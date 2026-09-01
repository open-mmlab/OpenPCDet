#####################
# Rotated 2D IoU / overlap for KITTI evaluation.
#
# The original implementation used a numba.cuda kernel. Recent numba releases
# combined with numpy >= 2.0 break that kernel's type inference ("Signature
# mismatch" raised from FixupArgs), and numpy cannot simply be pinned back on
# newer toolchains. It is replaced here by a thin PyTorch wrapper around
# OpenPCDet's own compiled boxes_overlap_bev_gpu op (pure C++/CUDA), which is
# already a hard dependency of the project, so no new dependency is introduced
# and the numba requirement disappears entirely.
#
# The rectangle geometry is reproduced exactly: OpenPCDet rotates local corners
# by R(heading) (CCW: [[cos,-sin],[sin,cos]]), while the original code rotated by
# [[cos,sin],[-sin,cos]] = R(-angle). Setting heading = -angle (with dx=x_d,
# dy=y_d) makes the two rectangles identical, so the intersection area returned
# here matches the original numba result numerically.
#####################
import numpy as np
import torch


def _to_bev7(boxes, device):
    # (N, 5) [cx, cy, dim_x, dim_y, angle] -> (N, 7) [x, y, z, dx, dy, dz, heading]
    n = boxes.shape[0]
    out = np.zeros((n, 7), dtype=np.float32)
    out[:, 0] = boxes[:, 0]      # center x
    out[:, 1] = boxes[:, 1]      # center y
    out[:, 3] = boxes[:, 2]      # dx  (original x_d)
    out[:, 4] = boxes[:, 3]      # dy  (original y_d)
    out[:, 5] = 1.0              # dz  (unused in BEV, must be > 0)
    out[:, 6] = -boxes[:, 4]     # heading = -angle  (match original R(-angle))
    return torch.from_numpy(out).to(device)


def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """Rotated 2D IoU / overlap used by the KITTI evaluator.

    Args:
        boxes:       (N, 5) [center_x, center_y, dim_x, dim_y, angle]
        query_boxes: (K, 5) same format
        criterion:   -1 -> IoU (inter / union)
                      0 -> inter / area(boxes)
                      1 -> inter / area(query_boxes)
                   other -> raw intersection area (used by d3_box_overlap)
        device_id:   CUDA device index to run on

    Returns:
        (N, K) float32 numpy array
    """
    from pcdet.ops.iou3d_nms import iou3d_nms_cuda

    boxes = np.ascontiguousarray(boxes, dtype=np.float32)
    query_boxes = np.ascontiguousarray(query_boxes, dtype=np.float32)
    N, K = boxes.shape[0], query_boxes.shape[0]
    if N == 0 or K == 0:
        return np.zeros((N, K), dtype=np.float32)

    device = torch.device('cuda', device_id)
    a = _to_bev7(boxes, device).contiguous()
    b = _to_bev7(query_boxes, device).contiguous()

    inter = torch.zeros((N, K), dtype=torch.float32, device=device)
    iou3d_nms_cuda.boxes_overlap_bev_gpu(a, b, inter)

    if criterion == -1:
        area_a = (a[:, 3] * a[:, 4]).view(-1, 1)
        area_b = (b[:, 3] * b[:, 4]).view(1, -1)
        ans = inter / torch.clamp(area_a + area_b - inter, min=1e-6)
    elif criterion == 0:
        area_a = (a[:, 3] * a[:, 4]).view(-1, 1)
        ans = inter / torch.clamp(area_a, min=1e-6)
    elif criterion == 1:
        area_b = (b[:, 3] * b[:, 4]).view(1, -1)
        ans = inter / torch.clamp(area_b, min=1e-6)
    else:
        ans = inter

    return ans.detach().cpu().numpy().astype(np.float32)
