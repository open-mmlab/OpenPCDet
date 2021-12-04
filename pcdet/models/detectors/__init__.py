from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .pyramid_rcnn_p import PyramidPoint
from .pyramid_rcnn_v import PyramidVoxel
from .pyramid_rcnn_pv import PyramidPointVoxel, PyramidPointVoxelPlus


__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'PyramidPoint': PyramidPoint,
    'PyramidVoxel': PyramidVoxel,
    'PyramidPointVoxel': PyramidPointVoxel,
    'PyramidPointVoxelPlus': PyramidPointVoxelPlus,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
