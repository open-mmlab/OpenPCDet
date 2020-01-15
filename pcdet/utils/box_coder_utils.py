import numpy as np
import torch
from . import common_utils


class ResidualCoder(object):
    def __init__(self, code_size=7):
        super().__init__()
        self.code_size = code_size

    @staticmethod
    def encode_np(boxes, anchors):
        """
        :param boxes: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        box_ndim = anchors.shape[-1]
        xa, ya, za, wa, la, ha, ra, *cas = np.split(anchors, box_ndim, axis=-1)
        xg, yg, zg, wg, lg, hg, rg, *cgs = np.split(boxes, box_ndim, axis=-1)

        # need to convert boxes to z-center format
        zg = zg + hg / 2
        za = za + ha / 2

        diagonal = np.sqrt(la ** 2 + wa ** 2)  # 4.3
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha  # 1.6
        lt = np.log(lg / la)
        wt = np.log(wg / wa)
        ht = np.log(hg / ha)
        rt = rg - ra
        cts = [g - a for g, a in zip(cgs, cas)]
        return np.concatenate([xt, yt, zt, wt, lt, ht, rt, *cts], axis=-1)

    @staticmethod
    def decode_np(box_encodings, anchors):
        """
        :param box_encodings: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        box_ndim = anchors.shape[-1]
        xa, ya, za, wa, la, ha, ra, *cas = np.split(anchors, box_ndim, axis=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = np.split(box_encodings, box_ndim, axis=-1)

        # need to convert box_encodings to z-bottom format
        za = za + ha / 2

        diagonal = np.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = np.exp(lt) * la
        wg = np.exp(wt) * wa
        hg = np.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return np.concatenate([xg, yg, zg, wg, lg, hg, rg, *cgs], axis=-1)

    @staticmethod
    def encode_torch(boxes, anchors):
        """
        :param boxes: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg, *cgs = torch.split(boxes, 1, dim=-1)

        za = za + ha / 2
        zg = zg + hg / 2

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
        rt = rg - ra

        cts = [g - a for g, a in zip(cgs, cas)]
        return torch.cat([xt, yt, zt, wt, lt, ht, rt, *cts], dim=-1)

    @staticmethod
    def decode_torch(box_encodings, anchors):
        """
        :param box_encodings: (N, 7 + ?) x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (N, 7 + ?)
        :return:
        """
        xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(box_encodings, 1, dim=-1)

        za = za + ha / 2

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra

        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)

    def decode_with_head_direction_torch(self, box_preds, anchors, dir_cls_preds,
                                         num_dir_bins, dir_offset, dir_limit_offset):
        """
        :param box_preds: (batch_size, N, 7 + ?), x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param anchors: (batch_size, N, 7 + ?), x, y, z, w, l, h, r, custom values, z is the box center in z-axis
        :param dir_cls_preds: (batch_size, H, W, num_anchors_per_locations*2)
        :return:
        """
        batch_box_preds = self.decode_torch(box_preds, anchors)

        if dir_cls_preds is not None:
            dir_cls_preds = dir_cls_preds.view(box_preds.shape[0], box_preds.shape[1], -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / num_dir_bins)
            dir_rot = common_utils.limit_period_torch(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        return batch_box_preds


class BinBasedCoder(object):
    def __init__(self, loc_scope, loc_bin_size, num_head_bin,
                 get_xz_fine=True, get_y_by_bin=False, loc_y_scope=0.5, loc_y_bin_size=0.25,
                 get_ry_fine=False, canonical_transform=False):
        super().__init__()
        self.loc_scope = loc_scope
        self.loc_bin_size = loc_bin_size
        self.num_head_bin = num_head_bin
        self.get_xz_fine = get_xz_fine
        self.get_y_by_bin = get_y_by_bin
        self.loc_y_scope = loc_y_scope
        self.loc_y_bin_size = loc_y_bin_size
        self.get_ry_fine = get_ry_fine
        self.canonical_transform = canonical_transform

    def decode_torch(self, pred_reg, roi_box3d, anchor_size):
        """
        decode in LiDAR coordinate
        :param pred_reg: (N, C)
        :param roi_box3d: (N, 7)
        :return:
        """
        anchor_size = anchor_size.to(roi_box3d.get_device())
        per_loc_bin_num = int(self.loc_scope / self.loc_bin_size) * 2
        loc_y_bin_num = int(self.loc_y_scope / self.loc_y_bin_size) * 2

        # recover xz localization
        x_bin_l, x_bin_r = 0, per_loc_bin_num
        z_bin_l, z_bin_r = per_loc_bin_num, per_loc_bin_num * 2
        start_offset = z_bin_r

        x_bin = torch.argmax(pred_reg[:, x_bin_l: x_bin_r], dim=1)
        z_bin = torch.argmax(pred_reg[:, z_bin_l: z_bin_r], dim=1)

        pos_x = x_bin.float() * self.loc_bin_size + self.loc_bin_size / 2 - self.loc_scope
        pos_z = z_bin.float() * self.loc_bin_size + self.loc_bin_size / 2 - self.loc_scope

        if self.get_xz_fine:
            x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
            z_res_l, z_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4
            start_offset = z_res_r

            x_res_norm = torch.gather(pred_reg[:, x_res_l: x_res_r], dim=1, index=x_bin.unsqueeze(dim=1)).squeeze(dim=1)
            z_res_norm = torch.gather(pred_reg[:, z_res_l: z_res_r], dim=1, index=z_bin.unsqueeze(dim=1)).squeeze(dim=1)
            x_res = x_res_norm * self.loc_bin_size
            z_res = z_res_norm * self.loc_bin_size

            pos_x += x_res
            pos_z += z_res

        # recover y localization
        if self.get_y_by_bin:
            y_bin_l, y_bin_r = start_offset, start_offset + loc_y_bin_num
            y_res_l, y_res_r = y_bin_r, y_bin_r + loc_y_bin_num
            start_offset = y_res_r

            y_bin = torch.argmax(pred_reg[:, y_bin_l: y_bin_r], dim=1)
            y_res_norm = torch.gather(pred_reg[:, y_res_l: y_res_r], dim=1, index=y_bin.unsqueeze(dim=1)).squeeze(dim=1)
            y_res = y_res_norm * self.loc_y_bin_size
            pos_y = y_bin.float() * self.loc_y_bin_size + self.loc_y_bin_size / 2 - self.loc_y_scope + y_res
            pos_y = pos_y + roi_box3d[:, 1]
        else:
            y_offset_l, y_offset_r = start_offset, start_offset + 1
            start_offset = y_offset_r

            pos_y = pred_reg[:, y_offset_l]

        # recover ry rotation
        ry_bin_l, ry_bin_r = start_offset, start_offset + self.num_head_bin
        ry_res_l, ry_res_r = ry_bin_r, ry_bin_r + self.num_head_bin

        ry_bin = torch.argmax(pred_reg[:, ry_bin_l: ry_bin_r], dim=1)
        ry_res_norm = torch.gather(pred_reg[:, ry_res_l: ry_res_r], dim=1, index=ry_bin.unsqueeze(dim=1)).squeeze(dim=1)
        if self.get_ry_fine:
            # divide pi/2 into several bins
            angle_per_class = (np.pi / 2) / self.num_head_bin
            ry_res = ry_res_norm * (angle_per_class / 2)
            ry = (ry_bin.float() * angle_per_class + angle_per_class / 2) + ry_res - np.pi / 4
        else:
            angle_per_class = (2 * np.pi) / self.num_head_bin
            ry_res = ry_res_norm * (angle_per_class / 2)

            # bin_center is (0, 30, 60, 90, 120, ..., 270, 300, 330)
            ry = (ry_bin.float() * angle_per_class + ry_res) % (2 * np.pi)
            ry[ry > np.pi] -= 2 * np.pi

        # recover size
        size_res_l, size_res_r = ry_res_r, ry_res_r + 3
        assert size_res_r == pred_reg.shape[1]

        size_res_norm = pred_reg[:, size_res_l: size_res_r]
        wlh = size_res_norm * anchor_size + anchor_size

        # shift to original coords
        roi_center = roi_box3d[:, 0:3]
        # Note: x, z, y, be consistent with get_reg_loss_lidar
        shift_ret_box3d = torch.cat((pos_x.view(-1, 1), pos_z.view(-1, 1), pos_y.view(-1, 1), wlh, ry.view(-1, 1)), dim=1)
        ret_box3d = shift_ret_box3d
        if self.canonical_transform and roi_box3d.shape[1] == 7:
            roi_ry = roi_box3d[:, 6]
            ret_box3d = rotate_pc_along_z_torch(shift_ret_box3d, (roi_ry + np.pi / 2))
            ret_box3d[:, 6] += roi_ry
        ret_box3d[:, 0:3] += roi_center

        return ret_box3d


def rotate_pc_along_z_torch(pc, rot_angle):
    """
    :param pc: (N, 3 + C)
    :param rot_angle: (N)
    :return:
    """
    cosa = torch.cos(rot_angle).view(-1, 1)
    sina = torch.sin(rot_angle).view(-1, 1)

    raw_1 = torch.cat([cosa, -sina], dim=1)
    raw_2 = torch.cat([sina, cosa], dim=1)
    R = torch.cat((raw_1.unsqueeze(dim=1), raw_2.unsqueeze(dim=1)), dim=1)  # (N, 2, 2)

    pc_temp = pc[:, 0:2].unsqueeze(dim=1)  # (N, 1, 2)

    pc[:, 0:2] = torch.matmul(pc_temp, R).squeeze(dim=1)
    return pc


def bbox2delta(proposals, gt, means=[0, 0, 0, 0], stds=[1, 1, 1, 1]):
    """
    :param proposals:
    :param gt:
    :param means:
    :param stds:
    :return:
    """
    assert proposals.size() == gt.size()

    proposals = proposals.float()
    gt = gt.float()
    px = (proposals[..., 0] + proposals[..., 2]) * 0.5
    py = (proposals[..., 1] + proposals[..., 3]) * 0.5
    pw = proposals[..., 2] - proposals[..., 0] + 1.0
    ph = proposals[..., 3] - proposals[..., 1] + 1.0

    gx = (gt[..., 0] + gt[..., 2]) * 0.5
    gy = (gt[..., 1] + gt[..., 3]) * 0.5
    gw = gt[..., 2] - gt[..., 0] + 1.0
    gh = gt[..., 3] - gt[..., 1] + 1.0

    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw / pw)
    dh = torch.log(gh / ph)
    deltas = torch.stack([dx, dy, dw, dh], dim=-1)

    means = deltas.new_tensor(means).unsqueeze(0)
    stds = deltas.new_tensor(stds).unsqueeze(0)
    deltas = deltas.sub_(means).div_(stds)

    return deltas


def delta2bbox(rois,
               deltas,
               means=[0, 0, 0, 0],
               stds=[1, 1, 1, 1],
               max_shape=None,
               wh_ratio_clip=16 / 1000):
    means = deltas.new_tensor(means).repeat(1, deltas.size(1) // 4)
    stds = deltas.new_tensor(stds).repeat(1, deltas.size(1) // 4)
    denorm_deltas = deltas * stds + means
    dx = denorm_deltas[:, 0::4]
    dy = denorm_deltas[:, 1::4]
    dw = denorm_deltas[:, 2::4]
    dh = denorm_deltas[:, 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = dw.clamp(min=-max_ratio, max=max_ratio)
    dh = dh.clamp(min=-max_ratio, max=max_ratio)
    px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
    py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
    pw = (rois[:, 2] - rois[:, 0] + 1.0).unsqueeze(1).expand_as(dw)
    ph = (rois[:, 3] - rois[:, 1] + 1.0).unsqueeze(1).expand_as(dh)
    gw = pw * dw.exp()
    gh = ph * dh.exp()
    gx = torch.addcmul(px, 1, pw, dx)  # gx = px + pw * dx
    gy = torch.addcmul(py, 1, ph, dy)  # gy = py + ph * dy
    x1 = gx - gw * 0.5 + 0.5
    y1 = gy - gh * 0.5 + 0.5
    x2 = gx + gw * 0.5 - 0.5
    y2 = gy + gh * 0.5 - 0.5
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1] - 1)
        y1 = y1.clamp(min=0, max=max_shape[0] - 1)
        x2 = x2.clamp(min=0, max=max_shape[1] - 1)
        y2 = y2.clamp(min=0, max=max_shape[0] - 1)
    bboxes = torch.stack([x1, y1, x2, y2], dim=-1).view_as(deltas)
    return bboxes


def decode_center_by_bin(center_pred, original_center, loc_scope, loc_bin_size):
    """
    :param center_pred: (N, C)
    :param original_center: (N, 3)
    :param loc_scope:
    :param loc_bin_size:
    :return:
    """
    per_loc_bin_num = int(loc_scope / loc_bin_size) * 2

    # recover xz localization
    x_bin_l, x_bin_r = 0, per_loc_bin_num
    y_bin_l, y_bin_r = per_loc_bin_num, per_loc_bin_num * 2
    x_res_l, x_res_r = per_loc_bin_num * 2, per_loc_bin_num * 3
    y_res_l, y_res_r = per_loc_bin_num * 3, per_loc_bin_num * 4

    x_bin = torch.argmax(center_pred[:, x_bin_l: x_bin_r], dim=1)
    y_bin = torch.argmax(center_pred[:, y_bin_l: y_bin_r], dim=1)
    x_res_norm = torch.gather(center_pred[:, x_res_l: x_res_r], dim=1, index=x_bin.unsqueeze(dim=1)).squeeze(dim=1)
    y_res_norm = torch.gather(center_pred[:, y_res_l: y_res_r], dim=1, index=y_bin.unsqueeze(dim=1)).squeeze(dim=1)
    x_res = x_res_norm * loc_bin_size
    y_res = y_res_norm * loc_bin_size

    pos_x = x_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope
    pos_y = y_bin.float() * loc_bin_size + loc_bin_size / 2 - loc_scope
    pos_x += x_res
    pos_y += y_res

    # recover y localization
    pos_z = center_pred[:, -1]

    center_res = torch.cat((pos_x.view(-1, 1), pos_y.view(-1, 1), pos_z.view(-1, 1)), dim=1)
    box_center = original_center + center_res
    return box_center


if __name__ == '__main__':
    A = ResidualCoder()

    roi = np.array([
        [1235.8512, 116.6956, 1241.0000, 313.0967],
        [  0.0000, 185.8831,   2.5533, 249.3910],
        [  0.0000, 277.3714,   0.0000, 374.0000],
        [ 53.1797, 153.9926, 217.4043, 254.9840],
        [ 62.8564, 150.7572, 228.6056, 254.1328],
        [ 51.9233, 155.1076, 210.8149, 257.6587],
        [214.3270, 168.2233, 286.9133, 202.7512],
        [397.9091,  86.6177, 692.8549, 198.1843],
        [312.8652, 157.6697, 517.4764, 247.0425],
        [  0.0000, 281.2801,   0.0000, 374.0000]])
    gt = np.array([
        [1210.4595, 88.2485, 1241.0000, 307.2321],
        [  0.0000, 188.9552,  12.8203, 248.2329],
        [  0.0000, 292.5262,   0.0000, 374.0000],
        [ 67.7120, 156.5728, 214.3772, 259.0395],
        [ 67.7120, 156.5728, 214.3772, 259.0395],
        [ 67.7120, 156.5728, 214.3772, 259.0395],
        [211.2084, 167.3544, 282.6477, 203.2750],
        [390.5256,  80.3167, 668.8065, 191.2320],
        [317.9528, 153.3572, 510.3288, 243.8742],
        [  0.0000, 292.5262,   0.0000, 374.0000]])

    roi = torch.from_numpy(roi)
    gt = torch.from_numpy(gt)

    import pdb
    pdb.set_trace()
    A = bbox2delta(roi, gt)



    import pdb
    pdb.set_trace()
