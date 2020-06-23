from .roi_head_template import RoIHeadTemplate
from .partA2_head import PartA2FCHead
from .pvrcnn_head import PVRCNNHead

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead
}
