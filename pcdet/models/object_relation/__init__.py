from .gnn import GNN
from .fc import CGNLNet
from .gnn_BADet import BARefiner
from .gnn_new import GNN_New

__all__ = {
    'GNN': GNN,
    'CGNLNet': CGNLNet,
    'GNN_BADET':BARefiner,
    'GNN_NEW': GNN_New,
}

def build_object_relation_module(model_cfg):
    model = __all__[model_cfg.NAME](
        model_cfg
    )
    return model