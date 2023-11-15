from .gnn import GNN
from .fc import CGNLNet

__all__ = {
    'GNN': GNN,
    'CGNLNet': CGNLNet
}

def build_object_relation_module(model_cfg):
    model = __all__[model_cfg.NAME](
        model_cfg
    )
    return model