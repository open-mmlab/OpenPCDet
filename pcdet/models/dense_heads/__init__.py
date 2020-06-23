from .anchor_head_template import AnchorHeadTemplate
from .anchor_head_single import AnchorHeadSingle
from .point_intra_part_head import PointIntraPartOffsetHead
from .point_head_simple import PointHeadSimple
from .anchor_head_multi import AnchorHeadMulti

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'AnchorHeadMulti': AnchorHeadMulti,
}
