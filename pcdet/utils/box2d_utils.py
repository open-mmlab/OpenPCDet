import torch


def area(box) -> torch.Tensor:
    """
    Computes the area of all the boxes.

    Returns:
        torch.Tensor: a vector with areas of each box.
    """
    area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    return area

# implementation from https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
# with slight modifications
def pairwise_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = area(boxes1)
    area2 = area(boxes2)

    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]
    del width_height

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou