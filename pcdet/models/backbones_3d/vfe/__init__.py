from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE
from .pointnet_pillar_vfe import PointNetPillarVFE
from .tpillar import TPillarVFE
from .vfe_template import VFETemplate

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'PointNetPillarVFE': PointNetPillarVFE,
    'TPillar': TPillarVFE
}
