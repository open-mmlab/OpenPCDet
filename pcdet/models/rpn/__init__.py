from .rpn_unet import UNetV0, UNetV2
from .rpn_backbone import BackBone8x
from .pillar_scatter import PointPillarsScatter

rpn_modules = {
    'UNetV2': UNetV2,
    'UNetV0': UNetV0,
    'BackBone8x': BackBone8x,
    'PointPillarsScatter': PointPillarsScatter
}
