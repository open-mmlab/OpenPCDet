import torch
import math
from ....ops.center_ops import center_ops_cuda

class CenterAssigner(object):
    def __init__(self, assigner_cfg, num_classes, no_log, grid_size, pc_range, voxel_size):
        """Return CenterNet training labels like heatmap, height, offset"""
        self.out_size_factor = assigner_cfg.out_size_factor
        self.num_classes = num_classes
        self.tasks = assigner_cfg.tasks
        self.dense_reg = assigner_cfg.dense_reg
        self.gaussian_overlap = assigner_cfg.gaussian_overlap
        self._max_objs = assigner_cfg.max_objs
        self._min_radius = assigner_cfg.min_radius
        self.class_to_idx = assigner_cfg.mapping
        self.grid_size = grid_size
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.no_log = no_log

    def gaussian_radius(self, height, width, min_overlap=0.5):
        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = math.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = math.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = math.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
        return min(r1, r2, r3)

    def gaussian_2d(self, shape, sigma = 1):
        m, n = [(ss - 1.) / 2. for ss in shape]
        mesh_m = torch.arange(start=-m, end=m+1, step=1, dtype=torch.float32)
        mesh_n = torch.arange(start=-n, end=n+1, step=1, dtype=torch.float32)
        y, x = torch.meshgrid([mesh_m, mesh_n])
        h = torch.exp(-(x * x + y * y) / (2 * sigma * sigma))
        eps = 1e-7
        h[h < eps * h.max()] = 0
        return h

    def draw_gaussian(self, heatmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self.gaussian_2d((diameter, diameter), sigma=diameter / 6)
        gaussian = gaussian.to(heatmap.device)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
            heatmap[y - top:y + bottom, x - left:x + right] = torch.stack([masked_heatmap, masked_gaussian * k], dim = 0).max(0)[0]
        return heatmap

    def limit_period(self, val, offset=0.5, period=math.pi):
        return val - math.floor(val / period + offset) * period

    def assign_targets_v1(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, C + cls)

        Returns:

        """
        max_objs = self._max_objs * self.dense_reg
        feature_map_size = self.grid_size[:2] // self.out_size_factor # grid_size WxHxD feature_map_size WxH

        batch_size = gt_boxes.shape[0]
        gt_classes = gt_boxes[:, :, -1] #begin from 1
        gt_boxes = gt_boxes[:, :, :-1]

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
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]
            cur_gt_classes = gt_classes[k][:cnt + 1].int()

            for task_id, task in enumerate(self.tasks):
                # heatmap size is supposed to be (cls_group, H, W)
                heatmap = torch.zeros((len(task.class_names), feature_map_size[1], feature_map_size[0]), dtype = torch.float32).to(cur_gt.device) #transpose ????
                gt_ind = torch.zeros(max_objs, dtype=torch.long).to(cur_gt.device)
                gt_mask = torch.zeros(max_objs, dtype=torch.bool).to(cur_gt.device)
                gt_cat = torch.zeros(max_objs, dtype=torch.long).to(cur_gt.device)
                gt_box_encoding = torch.zeros((max_objs, 10), dtype=torch.float32).to(cur_gt.device)

                cur_gts_of_task = []
                cur_classes_of_task = []
                class_offset = 0
                for class_name in task.class_names:
                    class_idx = self.class_to_idx[class_name]
                    class_mask = (cur_gt_classes == class_idx)
                    cur_gt_of_task = cur_gt[class_mask]
                    cur_class_of_task = cur_gt.new_full((cur_gt_of_task.shape[0],), class_offset).long()
                    cur_gts_of_task.append(cur_gt_of_task)
                    cur_classes_of_task.append(cur_class_of_task)
                    class_offset += 1
                cur_gts_of_task = torch.cat(cur_gts_of_task, dim = 0)
                cur_classes_of_task = torch.cat(cur_classes_of_task, dim = 0)

                num_boxes_of_task = cur_gts_of_task.shape[0]
                for i in range(num_boxes_of_task):
                    cat = cur_classes_of_task[i]
                    x, y, z, w, l, h, r, vx, vy = cur_gts_of_task[i]
                    #r -> [-pi, pi]
                    r = self.limit_period(r, offset=0.5, period=math.pi * 2)
                    w, l = w / self.voxel_size[0] / self.out_size_factor, l / self.voxel_size[1] / self.out_size_factor
                    radius = self.gaussian_radius(l, w, min_overlap=self.gaussian_overlap)
                    radius = max(self._min_radius, int(radius))

                    coor_x = (x - self.pc_range[0]) / self.voxel_size[0] / self.out_size_factor
                    coor_y = (y - self.pc_range[1]) / self.voxel_size[1] / self.out_size_factor
                    ct_ft = torch.tensor([coor_x, coor_y], dtype = torch.float32)
                    ct_int = ct_ft.int() #float to int conversion torch/np

                    if not (0 <= ct_int[0] < feature_map_size[0] and 0 <= ct_int[1] < feature_map_size[1]):
                        continue

                    self.draw_gaussian(heatmap[cat], ct_int, radius) #pass functions

                    gt_cat[i] = cat
                    gt_mask[i] = 1
                    gt_ind[i] = ct_int[1] * feature_map_size[0] + ct_int[0]
                    assert gt_ind[i] < feature_map_size[0] * feature_map_size[1]
                    # Note that w,l has been modified, so in box encoding we use original w,l,h
                    if not self.no_log:
                        gt_box_encoding[i] = torch.tensor([ct_ft[0] - ct_int[0],
                                                           ct_ft[1] - ct_int[1],
                                                           z,
                                                           math.log(cur_gts_of_task[i,3]),
                                                           math.log(cur_gts_of_task[i,4]),
                                                           math.log(cur_gts_of_task[i,5]),
                                                           math.sin(r),
                                                           math.cos(r),
                                                           vx,
                                                           vy
                                                           ], dtype=torch.float32).to(gt_box_encoding.device)
                    else:
                        gt_box_encoding[i] = torch.tensor([ct_ft[0] - ct_int[0],
                                                           ct_ft[1] - ct_int[1],
                                                           z,
                                                           cur_gts_of_task[i,3],
                                                           cur_gts_of_task[i,4],
                                                           cur_gts_of_task[i,5],
                                                           math.sin(r),
                                                           math.cos(r),
                                                           vx,
                                                           vy
                                                           ], dtype=torch.float32).to(gt_box_encoding.device)

                heatmaps[task_id].append(heatmap)
                gt_inds[task_id].append(gt_ind)
                gt_cats[task_id].append(gt_cat)
                gt_masks[task_id].append(gt_mask)
                gt_box_encodings[task_id].append(gt_box_encoding)

        for task_id, tasks in enumerate(self.tasks):
            heatmaps[task_id] = torch.stack(heatmaps[task_id], dim = 0).contiguous()
            gt_inds[task_id] = torch.stack(gt_inds[task_id], dim = 0).contiguous()
            gt_masks[task_id] = torch.stack(gt_masks[task_id], dim = 0).contiguous()
            gt_cats[task_id] = torch.stack(gt_cats[task_id], dim = 0).contiguous()
            gt_box_encodings[task_id] = torch.stack(gt_box_encodings[task_id], dim = 0).contiguous()

        target_dict = {
            'heatmap': heatmaps,
            'ind': gt_inds,
            'mask': gt_masks,
            'cat': gt_cats,
            'box_encoding': gt_box_encodings
        }
        return target_dict

    def assign_targets_v2(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, C + cls)
        Returns:
        """
        max_objs = self._max_objs * self.dense_reg
        feature_map_size = self.grid_size[:2] // self.out_size_factor # grid_size WxHxD feature_map_size WxH
        batch_size = gt_boxes.shape[0]
        code_size = gt_boxes.shape[2] #cls -> sin/cos
        num_classes = self.num_classes
        assert gt_boxes[:, :, -1].max().item() <= num_classes, "labels must match, found {}".format(gt_boxes[:, :, -1].max().item())

        heatmaps = {}
        gt_inds = {}
        gt_masks = {}
        gt_box_encodings = {}
        gt_cats = {}

        heatmap = torch.zeros((batch_size, num_classes, feature_map_size[1], feature_map_size[0]), dtype = torch.float32).to(gt_boxes.device)
        gt_ind = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_mask = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_cat = torch.zeros((batch_size, num_classes, max_objs), dtype = torch.int32).to(gt_boxes.device)
        gt_cnt = torch.zeros((batch_size, num_classes), dtype = torch.int32).to(gt_boxes.device)
        gt_box_encoding = torch.zeros((batch_size, num_classes, max_objs, code_size), dtype = torch.float32).to(gt_boxes.device)

        center_ops_cuda.draw_center_gpu(gt_boxes, heatmap, gt_ind, gt_mask, gt_cat, gt_box_encoding, gt_cnt,
                        self._min_radius, self.voxel_size[0], self.voxel_size[1], self.pc_range[0], self.pc_range[1],
                        self.out_size_factor, self.gaussian_overlap)

        offset = 0
        for task_id, task in enumerate(self.tasks):
            end = offset + len(task.class_names)
            heatmap_of_task = heatmap[:, offset:end, :, :]
            gt_ind_of_task = gt_ind[:, offset:end, :].reshape(batch_size, -1)
            gt_mask_of_task = gt_mask[:, offset:end, :].reshape(batch_size, -1)
            gt_cat_of_task = gt_cat[:, offset:end, :].reshape(batch_size, -1) - (offset + 1) # cat begin from 1
            gt_box_encoding_of_task = gt_box_encoding[:, offset:end, :, :].reshape(batch_size, -1, code_size)
            gt_ind_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_mask_merged = torch.zeros((batch_size, max_objs), dtype=torch.int32).to(gt_boxes.device)
            gt_cat_merged = torch.zeros((batch_size, max_objs), dtype = torch.int32).to(gt_boxes.device)
            gt_box_encoding_merged = torch.zeros((batch_size, max_objs, code_size), dtype=torch.float32).to(gt_boxes.device)
            offset = end
            for i in range(batch_size):
                mask = gt_mask_of_task[i] == 1
                mask_range = mask.sum().item()
                assert mask_range <= max_objs
                gt_mask_merged[i, :mask_range] = gt_mask_of_task[i, mask]
                gt_ind_merged[i, :mask_range] = gt_ind_of_task[i, mask]
                gt_cat_merged[i, :mask_range] = gt_cat_of_task[i, mask]
                gt_box_encoding_merged[i, :mask_range, :] = gt_box_encoding_of_task[i, mask, :]
                # only perform log on valid gt_box_encoding
                if not self.no_log:
                    gt_box_encoding_merged[i, :mask_range, 3:6] = torch.log(gt_box_encoding_merged[i, :mask_range, 3:6]) # log(wlh)

            heatmaps[task_id] = heatmap_of_task
            gt_inds[task_id] = gt_ind_merged.long()
            gt_masks[task_id] = gt_mask_merged.bool()
            gt_cats[task_id] = gt_cat_merged.long()
            gt_box_encodings[task_id] = gt_box_encoding_merged

        target_dict = {
            'heatmap': heatmaps,
            'ind': gt_inds,
            'mask': gt_masks,
            'cat': gt_cats,
            'box_encoding': gt_box_encodings
        }
        return target_dict