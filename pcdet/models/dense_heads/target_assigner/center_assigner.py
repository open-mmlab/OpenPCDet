import torch
import math
import numpy as np


class CenterAssigner(object):
    def __init__(self, assigner_cfg, num_class, no_log, grid_size, point_cloud_range, voxel_size):
        """Return CenterNet training labels likt heatmap, height, offset"""
        self.assigner_cfg = assigner_cfg
        self.dense_reg = assigner_cfg.dense_reg
        self.out_size_factor = assigner_cfg.out_size_factor
        self.tasks = assigner_cfg.tasks
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.no_log = no_log

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
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 - sq1) / (2 * a1)

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 - sq2) / (2 * a2)

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)
        return min(r1, r2, r3)

    def gaussian_2d(self, shape, sigma=1):
        """Generate gaussian map.

        Args:
            shape (Tuple[int]): Shape of the map.
            sigma (float): Sigma to generate gaussian map.
                Defaults to 1.

        Returns:
            np.ndarray: Generated gaussian map.
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        # limit extreme small number
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
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
        # 将高斯分布结果约束在边界内
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
        feature_map_size = self.grid_size[:2] // self.out_size_factor

        batch_size = gt_boxes.shape[0]
        gt_classes = gt_boxes[:,:,-1] # last layer
        gt_boxes = gt_boxes[:,:,:-1] # except for last layer

        heatmaps = {}
        gt_inds = {}
        gt_masks = {}
        gt_box_encodings = {}
        gt_cats = {}
        for task_id,task in enumerate(self.tasks):
            heatmaps[task_id] = []
            gt_inds[task_id] = []
            gt_masks[task_id] = []
            gt_box_encodings[task_id] = []
            gt_cats[task_id] = []

        for k in range(batch_size):
            # TODO: I dont understand this part
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()

            for task_id, task in enumerate(self.tasks):
                # heatmap size (cls_group, H, W), why size1,size0?
                heatmap = torch.zeros(
                    (len(task.class_names),
                     feature_map_size[1],
                     feature_map_size[0]),dtype=torch.float32,device=cur_gt.device)
                gt_ind = torch.zeros(max_objs,dtype=torch.long,device=cur_gt.device)
                gt_mask = torch.zeros(max_objs, dtype=torch.long, device=cur_gt.device)
                gt_cat = torch.zeros(max_objs, dtype=torch.long, device=cur_gt.device)
                gt_box_encoding = torch.zeros(max_objs, dtype=torch.long, device=cur_gt.device)

                cur_gts_of_task = []
                cur_classes_of_task = []
                class_offset = 0
                for class_name in task.class_names:
                    class_idx = self.class_to_idx[class_name]
                    class_mask = (gt_classes ==class_idx)
                    cur_gt_of_task = cur_gt[class_mask]
                    # fill will class offset number
                    cur_class_of_task = cur_gt.new_full((cur_gt_of_task.shape[0],),class_offset).long()
                    cur_gts_of_task.append(cur_gt_of_task)
                    cur_classes_of_task.append(cur_class_of_task)
                    class_offset += 1
                cur_gts_of_task = torch.cat(cur_gts_of_task,dim=0)
                cur_classes_of_task = torch.cat(cur_classes_of_task,dim=0)

                num_boxes_of_task = cur_gts_of_task.shape[0]




        for task_id,task in enumerate(self.tasks):
            heatmaps[task_id] = torch.stack(heatmaps[task_id],dim = 0).contiguous()
            gt_inds[task_id] = torch.stack(gt_inds[task_id], dim=0).contiguous()
            gt_masks[task_id] = torch.stack(gt_masks[task_id], dim=0).contiguous()
            gt_box_encodings[task_id] = torch.stack(gt_box_encodings[task_id], dim=0).contiguous()
            gt_cats[task_id] = torch.stack(gt_cats[task_id], dim=0).contiguous()

        target_dict = {
            'heatmap': heatmaps,
            'ind': gt_inds,
            'mask': gt_masks,
            'cat': gt_cats,
            'box_encoding': gt_box_encodings
        }

        return target_dict

    def get_targets(self, gt_bboxes_3d, gt_labels_3d):
        """Generate targets.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.

        Returns:
            Returns:
                tuple[list[torch.Tensor]]: Tuple of target including \
                    the following results in order.

                    - list[torch.Tensor]: Heatmap scores.
                    - list[torch.Tensor]: Ground truth boxes.
                    - list[torch.Tensor]: Indexes indicating the \
                        position of the valid boxes.
                    - list[torch.Tensor]: Masks indicating which \
                        boxes are valid.
        """
        heatmaps, anno_boxes, inds, masks = multi_apply(
            self.get_targets_single, gt_bboxes_3d, gt_labels_3d)
        # transpose heatmaps, because the dimension of tensors in each task is
        # different, we have to use numpy instead of torch to do the transpose.
        heatmaps = np.array(heatmaps).transpose(1, 0).tolist()
        heatmaps = [torch.stack(hms_) for hms_ in heatmaps]
        # transpose anno_boxes
        anno_boxes = np.array(anno_boxes).transpose(1, 0).tolist()
        anno_boxes = [torch.stack(anno_boxes_) for anno_boxes_ in anno_boxes]
        # transpose inds
        inds = np.array(inds).transpose(1, 0).tolist()
        inds = [torch.stack(inds_) for inds_ in inds]
        # transpose inds
        masks = np.array(masks).transpose(1, 0).tolist()
        masks = [torch.stack(masks_) for masks_ in masks]
        return heatmaps, anno_boxes, inds, masks

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        device = gt_labels_3d.device
        gt_bboxes_3d = torch.cat(
            (gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]),
            dim=1).to(device)
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(torch.cat(task_box, axis=0).to(device))
            task_classes.append(torch.cat(task_class).long().to(device))
            flag2 += len(mask)
        draw_gaussian = self.draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]))

            anno_box = gt_bboxes_3d.new_zeros((max_objs, 10),
                                              dtype=torch.float32)

            ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64)
            mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = self.gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                                     x - pc_range[0]
                             ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                                     y - pc_range[1]
                             ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert (y * feature_map_size[0] + x <
                            feature_map_size[0] * feature_map_size[1])

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device=device),
                        z.unsqueeze(0), box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0),
                        vx.unsqueeze(0),
                        vy.unsqueeze(0)
                    ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks