import torch
import math
import numpy as np
import pdb


class CenterAssigner(object):
    def __init__(self, assigner_cfg, num_class, no_log, grid_size, point_cloud_range, voxel_size, dataset):
        """Return CenterNet training labels like heatmap, height, offset"""
        self.assigner_cfg = assigner_cfg
        self.dense_reg = assigner_cfg.dense_reg
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.no_log = no_log
        self.dataset = dataset

        self.num_class = num_class
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.class_to_idx = assigner_cfg.mapping

    def gaussian_radius(self, det_size, min_overlap=0.5):
        """Get radius of gaussian.
        details in https://zhuanlan.zhihu.com/p/96856635
        https://github.com/princeton-vl/CornerNet/issues/110
        https://github.com/princeton-vl/CornerNet/commit/3e71377b45098f9cea26d5a39de0138174c90d49

        Args:
            det_size (tuple[torch.Tensor]): Size of the detection result.
            min_overlap (float): Gaussian_overlap. Defaults to 0.5.

        Returns:
            torch.Tensor: Computed radius.
        """

        height, width = det_size

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 - sq1) / (2 * a1)

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 - sq2) / (2 * a2)

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)
        return min(r1, r2, r3)

    def gaussian_2d(self, shape, sigma=1):
        """Generate gaussian map.

        Args:
            shape (Tuple[int]): Shape of the map.
            sigma (float): Sigma to generate gaussian map.
                Defaults to 1.

        Returns:
            math.ndarray: Generated gaussian map.
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        # limit extreme small number
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        # h = torch.from_numpy(h)
        # if torch.cuda.is_available():
        #     h = h.cuda()
        return h

    def draw_heatmap_gaussian(self, heatmap, center, radius, k=1):
        """Get gaussian masked heatmap.

        Args:
            heatmap (torch.Tensor): Heatmap to be masked.
            center (torch.Tensor): Center coord of the heatmap.
            radius (int): Radius of gausian.
            K (int): Multiple of masked_gaussian. Defaults to 1.

        Returns:
            torch.Tensor: Masked heatmap.
        """
        # 得到直径
        diameter = 2 * radius + 1
        gaussian = self.gaussian_2d((diameter, diameter), sigma=diameter / 6)
        # sigma 是一个与直径相关的参数
        # 一个圆对应内切正方形的高斯分布

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        # 对边界进行约束，防止越界，x - right >= 0, x + right <= w
        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        # 选择对应区域
        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        # 将高斯分布结果约束在边界内, gaussian 格式为numpy, 转为tensor
        masked_gaussian = torch.from_numpy(
            gaussian[radius - top:radius + bottom,
            radius - left:radius + right]).to(heatmap.device,
                                              torch.float32)
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        # 将高斯分布覆盖到heatmap 上，相当于不断的在heatmap 基础上添加关键点的高斯，
        # 即同一种类型的框会在一个heatmap 某一个类别通道上面上面不断添加。
        # 最终通过函数总体的for 循环，相当于不断将目标画到heatmap
        return heatmap

    def assign_targets(self, gt_boxes):
        """

        Args:
            gt_boxes: (B, M, C + cls)

        Returns:

        """
        max_objs = self._max_objs * self.dense_reg
        # grid size shape (W, H, D), feature map shape (W/S, H/S)
        feature_map_size = [(i - 1) // self.out_size_factor + 1 for i in self.grid_size[:2]]

        batch_size = gt_boxes.shape[0]
        gt_classes = gt_boxes[:, :, -1]  # last layer
        gt_boxes = gt_boxes[:, :, :-1]  # except for last layer

        heatmaps = {}
        gt_inds = {}
        gt_masks = {}
        gt_box_encodings = {}
        gt_cats = {}
        for task_id, task in enumerate(self.tasks):
            heatmaps[task_id] = []
            gt_inds[task_id] = []
            gt_masks[task_id] = []
            gt_box_encodings[task_id] = []
            gt_cats[task_id] = []

        for k in range(batch_size):
            # TODO: I dont understand this part
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1  # M gt boxes, second dim
            # 看看是不是填充的0，只看非0的箱子
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()

            for task_id, task in enumerate(self.tasks):
                # heatmap size (cls_group, H, W), why size1,size0?
                heatmap = torch.zeros(
                    (len(task.class_names),
                     feature_map_size[1],
                     feature_map_size[0]), dtype=torch.float32, device=cur_gt.device)
                gt_ind = torch.zeros(max_objs, dtype=torch.long, device=cur_gt.device)
                gt_mask = torch.zeros(max_objs, dtype=torch.bool, device=cur_gt.device)
                gt_cat = torch.zeros(max_objs, dtype=torch.long, device=cur_gt.device)
                if self.dataset == 'nuscenes':
                    gt_box_encoding = torch.zeros((max_objs, 10), dtype=torch.float32, device=cur_gt.device)
                elif self.dataset == 'waymo':
                    gt_box_encoding = torch.zeros((max_objs, 8), dtype=torch.float32, device=cur_gt.device)
                else:
                    raise NotImplementedError("Only Support KITTI and nuScene for Now!")

                cur_gts_of_task = []
                cur_classes_of_task = []
                # offset start from 0, means the index of heatmap
                class_offset = 0
                for class_name in task.class_names:
                    class_idx = self.class_to_idx[class_name]
                    # some question about mask in this part
                    # in dataset.py, cls_idx start from 1
                    class_mask = (cur_gt_classes == class_idx)
                    cur_gt_of_task = cur_gt[class_mask]
                    # fill will class offset number
                    cur_class_of_task = cur_gt.new_full((cur_gt_of_task.shape[0],), class_offset).long()
                    cur_gts_of_task.append(cur_gt_of_task)
                    cur_classes_of_task.append(cur_class_of_task)
                    class_offset += 1
                cur_gts_of_task = torch.cat(cur_gts_of_task, dim=0)
                cur_classes_of_task = torch.cat(cur_classes_of_task, dim=0)

                num_boxes_of_task = cur_gts_of_task.shape[0]
                for i in range(num_boxes_of_task):
                    cat = cur_classes_of_task[i]  # category
                    # TODO: different datasets have different format
                    x, y, z, w, l, h, r = cur_gts_of_task[i][:7]
                    if self.dataset == 'nuscenes':
                        x, y, z, w, l, h, r, vx, vy = cur_gts_of_task[i]
                    elif self.dataset == 'waymo':
                        x, y, z, w, l, h, r = cur_gts_of_task[i]
                    # -pi < r - 2 * pi * k < pi, find k:int
                    r = r - math.floor(r / (2 * math.pi) + 0.5) * 2 * math.pi
                    w, l = w / self.voxel_size[0] / self.out_size_factor, l / self.voxel_size[1] / self.out_size_factor
                    if w > 0 and l > 0:
                        radius = self.gaussian_radius((l, w), min_overlap=self.gaussian_overlap)
                        radius = max(self._min_radius, int(radius))

                        # 坐标系转换
                        # be really careful for the coordinate system of your box annotation.
                        coor_x, coor_y = (x - self.point_cloud_range[0]) / self.voxel_size[0] / self.out_size_factor, \
                                         (y - self.point_cloud_range[1]) / self.voxel_size[1] / self.out_size_factor

                        ct_ft = torch.tensor(
                            [coor_x, coor_y], dtype=torch.float32)
                        ct_int = ct_ft.int()

                        # throw out not in range objects to avoid out of array area when creating the heatmap
                        if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                            continue

                        self.draw_heatmap_gaussian(heatmap[cat], ct_int, radius)

                        new_idx = i
                        x, y = ct_int[0], ct_int[1]

                        if not (y * feature_map_size[0] + x < feature_map_size[0] * feature_map_size[1]):
                            # a double check, should never happen
                            print(x, y, y * feature_map_size[0] + x)
                            assert False

                        gt_cat[new_idx] = cat
                        gt_ind[new_idx] = y * feature_map_size[0] + x
                        gt_mask[new_idx] = 1

                        # w,l has been modified, so in box encoding, we use original w,l,h
                        # TODO
                        if not self.no_log:
                            w, l, h = math.log(cur_gts_of_task[i, 3]), math.log(cur_gts_of_task[i, 4]), math.log(
                                cur_gts_of_task[i, 5])
                        else:
                            w, l, h = cur_gts_of_task[i, 3], cur_gts_of_task[i, 4], cur_gts_of_task[i, 5]
                        if self.dataset == 'nuscenes':
                            gt_box_encoding[i] = torch.tensor([ct_ft[0] - ct_int[0],
                                                               ct_ft[1] - ct_int[1],
                                                               z, w, l, h,
                                                               math.sin(r), math.cos(r),
                                                               vx, vy
                                                               ], dtype=torch.float32, device=gt_box_encoding.device)
                        elif self.dataset == 'waymo':
                            gt_box_encoding[i] = torch.tensor([ct_ft[0] - ct_int[0],
                                                               ct_ft[1] - ct_int[1],
                                                               z, w, l, h,
                                                               math.sin(r), math.cos(r)
                                                               ], dtype=torch.float32, device=gt_box_encoding.device)
                        else:
                            raise NotImplementedError("Only Support KITTI and nuScene for Now!")

                heatmaps[task_id].append(heatmap)
                gt_inds[task_id].append(gt_ind)
                gt_cats[task_id].append(gt_cat)
                gt_masks[task_id].append(gt_mask)
                gt_box_encodings[task_id].append(gt_box_encoding)

        for task_id, task in enumerate(self.tasks):
            heatmaps[task_id] = torch.stack(heatmaps[task_id], dim=0).contiguous()
            gt_inds[task_id] = torch.stack(gt_inds[task_id], dim=0).contiguous()
            gt_masks[task_id] = torch.stack(gt_masks[task_id], dim=0).contiguous()
            gt_box_encodings[task_id] = torch.stack(gt_box_encodings[task_id], dim=0).contiguous()
            gt_cats[task_id] = torch.stack(gt_cats[task_id], dim=0).contiguous()

        target_dict = {
            'heatmap': heatmaps,
            'ind': gt_inds,
            'mask': gt_masks,
            # cat seems no use in this task
            'cat': gt_cats,
            'box_encoding': gt_box_encodings
        }

        return target_dict
