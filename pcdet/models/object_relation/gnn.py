import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from .utils import build_mlp
# import torch_scatter

# custom implementation for EdgeConv
# class EdgeWeightingNetwork(nn.Module):
#     def __init__(self, dimension):
#         super(EdgeWeightingNetwork, self).__init__()
#         self.fc = nn.Linear(dimension*2, dimension)
        
#     def forward(self, x, edge_index):
#         from_node, to_node = edge_index
#         x_i, x_j = x[from_node], x[to_node]
#         e_ij = torch.cat((x_j - x_i, x_i), dim=-1)
#         e_ij = self.fc(e_ij)
#         e_ij = F.relu(e_ij)
#         # why from node here?????
#         out, _ = torch_scatter.scatter_max(e_ij, from_node, dim=0)
#         return out

# similar to EdgeConv https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.EdgeConv.html
class EdgeConv(tg.nn.MessagePassing):
    def __init__(self, dim_in, dim_out, drop_out=None, skip_connection=False):
        super(EdgeConv, self).__init__(aggr='max')
        self.skip_connection = skip_connection
        self.mlp = build_mlp(dim_in, [dim_out], activation="ReLU", bn=True, drop_out=drop_out)
        self.batch_norm = nn.BatchNorm1d(dim_out)

    def forward(self, x, edge_index, edge_attr=None):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.batch_norm(out)
        if self.skip_connection:
            return out + x
        return out  

    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr is not None:
            x = torch.cat((x_j - x_i, x_i, edge_attr), dim=-1)
        else:
            x = torch.cat((x_j - x_i, x_i), dim=-1)
        x = self.mlp(x)
        return x


class GNN(nn.Module):
    def __init__(self, object_relation_cfg, number_classes=3, pooled_feature_dim=256):
        super(GNN, self).__init__()
        self.graph_cfg = object_relation_cfg.GRAPH
        self.graph_conv = object_relation_cfg.GRAPH.CONV
        self.gnn_layers = object_relation_cfg.LAYERS
        self.in_between_layers = object_relation_cfg.IN_BETWEEN_MLP
        # self.in_between_layers = object_relation_cfg.IN_BETWEEN_MLP if 'IN_BETWEEN_MLP' in  object_relation_cfg else None
        self.global_information = object_relation_cfg.GLOBAL_INFORMATION
        # self.global_information = object_relation_cfg.GLOBAL_INFORMATION if 'GLOBAL_INFORMATION' in object_relation_cfg  else None
        self.number_classes = number_classes
        self.drop_out = object_relation_cfg.DP_RATIO
        self.skip_connection = object_relation_cfg.SKIP_CONNECTION
        self.pooled_feature_dim = pooled_feature_dim

        if self.global_information:
            global_mlp_input_dim = pooled_feature_dim + 7 if self.global_information.CONCATENATED else 7
            self.global_info_mlp = build_mlp(global_mlp_input_dim, self.global_information.MLP_LAYERS, activation='ReLU', bn=True, drop_out=self.drop_out)
        
        if not self.global_information:
            gnn_input_dim = pooled_feature_dim
        else:
            gnn_input_dim = self.global_mlp[-1] if self.global_information.CONCATENATED else (self.global_mlp[-1] + self.pooled_feature_dim)
        
        conv_layer_list = []
        for i in range(len(self.gnn_layers)):
            curr_conv_layer_list = []
            if i == 0:
                input_dim = gnn_input_dim
            else:
                input_dim = self.gnn_layers[i-1]

            edge_dim = (7 if self.graph_conv.EDGE_EMBEDDING else 0)
            if self.graph_conv.NAME == "EdgeConv":
                curr_conv_layer_list.append(EdgeConv(2*input_dim+edge_dim, self.gnn_layers[i], drop_out=self.drop_out,  skip_connection=self.graph_conv.SKIP_CONNECTION))
            elif self.graph_conv.NAME == "GATConv":
                # layer according to tg example: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/gat.py
                curr_conv_layer_list.append(nn.Dropout(p=self.drop_out))
                curr_conv_layer_list.append(tg.nn.GATConv(input_dim, self.gnn_layers[i], self.graph_conv.HEADS, dropout=self.drop_out, edge_dim=edge_dim, concat=False))
                curr_conv_layer_list.append(nn.ELU())
            if self.in_between_layers:
                curr_mlp = build_mlp(self.gnn_layers[i], [self.in_between_layers[i]], activation="ReLU", bn=True, drop_out=True)
                curr_conv_layer_list.append(curr_mlp)
            conv_layer_list.append(nn.ModuleList(curr_conv_layer_list))
        self.gnn = nn.ModuleList(conv_layer_list)
        self.init_weights()


    def init_weights(self, weight_init='xavier'):
        if weight_init == 'xavier':
            init_func = nn.init.xavier_uniform_
        for seq in self.gnn:
            for n in seq:
                if isinstance(n, EdgeConv):
                    for m in n.mlp:
                        if isinstance(m, nn.Linear):
                            init_func(m.weight)
                elif isinstance(n, nn.Sequential):
                    for m in n:
                        if isinstance(m, nn.Linear):
                            init_func(m.weight)
                elif isinstance(n, tg.nn.GATConv):
                    # automatically initialized
                    continue
                else:
                    continue
        if self.global_information:
            for m in self.global_info_mlp:
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
    
    def forward(self, batch_dict):
        (B, N, C) = batch_dict['pooled_features'].shape

        # BxNx7
        proposal_boxes = batch_dict['rois'].view(B*N,7)
        # BxN
        proposal_labels = batch_dict['roi_labels'].view(B*N)
        # BxNxC
        pooled_features = batch_dict['pooled_features'].view(B*N,C)
        

        if self.global_information:
            if self.global_information.CONCATENATED:
                pooled_features_with_global_info = torch.cat([pooled_features, proposal_boxes], dim=1)
                pooled_features = self.global_info_mlp(pooled_features_with_global_info)
            else:
                embedded_global_information = self.global_info_mlp(proposal_boxes)
                pooled_features = torch.cat([pooled_features, embedded_global_information], dim=1)
        
        if self.graph_cfg.SPACE == 'R3':
            assert self.graph_cfg.DYNAMIC == False, 'Distance should be measured in feature space if the graph is created dynamically'
            edge_index = self.get_edges(proposal_boxes[:,:3], proposal_labels, (B, N, C))
        elif self.graph_cfg.SPACE == 'Feature':
            edge_index = self.get_edges(pooled_features, proposal_labels, (B, N, C))
        else:
            raise NotImplemented('Distance space was {} but should be R3 or FEATURE'.format(self.graph_cfg.SPACE))
        
        batch_dict['gnn_edges'] = edge_index

        edge_attr = None
        if self.graph_conv.EDGE_EMBEDDING:
            from_node, to_node = edge_index
            edge_attr = proposal_boxes[from_node] - proposal_boxes[to_node]
        
        gnn_features = [pooled_features]
        x = pooled_features
        for module_list in self.gnn:
            for module in module_list:
                if isinstance(module, (EdgeConv, tg.nn.GATConv)):
                    x = module(x, edge_index, edge_attr=edge_attr)
                else:
                    x = module(x)
            gnn_features.append(x)
            if self.graph_cfg.DYNAMIC:
                edge_index = self.get_edges(x, proposal_labels, (B, N, None))
                if edge_attr is not None:
                    from_node, to_node = edge_index
                    edge_attr = proposal_boxes[from_node] - proposal_boxes[to_node]

        if self.skip_connection:
            batch_dict['related_features'] = torch.cat(gnn_features, dim=-1)
        else:
            batch_dict['related_features'] = x

        return batch_dict

    def get_edges(self, edge_generating_tensor, proposal_labels, shape):
        B, N, _ = shape
        f = getattr(tg.nn, self.graph_cfg.NAME)
        a = (self.graph_cfg.RADIUS if self.graph_cfg.NAME == 'radius_graph' else self.graph_cfg.K)
        batch_vector = torch.arange(B, device=edge_generating_tensor.device).repeat_interleave(N)
        
        if self.graph_cfg.CONNECT_ONLY_SAME_CLASS:
            final_edge_indices = []
            for predicted_class in range(1, self.number_classes + 1):
                label_indices = torch.where(proposal_labels == predicted_class)[0]
                label_edge_generating_tensor = edge_generating_tensor[label_indices]
                label_batch_vector = batch_vector[label_indices]
                
                label_edge_index = f(label_edge_generating_tensor, a, batch=label_batch_vector, loop=False)
                label_edge_index[0] = label_indices[label_edge_index[0]]
                label_edge_index[1] = label_indices[label_edge_index[1]]
                final_edge_indices.append(label_edge_index)
            edge_index = torch.cat(final_edge_indices, dim=-1)
        else:
            edge_index = f(edge_generating_tensor, a, batch=batch_vector, loop=False)
        return edge_index


if __name__ == '__main__':
    from easydict import EasyDict as edict
    rois = torch.tensor([
        [[0, 0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3, 3]]
    ], dtype=torch.float32)
    # 1x3x512
    pooled_features = torch.rand((1, 3, 256), dtype=torch.float32)
    proposal_labels = torch.tensor([
        [0, 0, 1]
    ], dtype=torch.int64)

    batch_dict = {
        'rois': rois,  # Random positions for 10 batches of 100 proposals each
        'pooled_features': pooled_features,  # Random 16-dimensional features for 10 batches of 100 proposals each
        'roi_labels': proposal_labels  # Random labels for 10 batches of 100 proposals each
    }

    cfg = edict({
        'GRAPH': {
            'NAME': 'radius_graph',
            'RADIUS': 3,
            'CONNECT_ONLY_SAME_CLASS': True
        },
        'LAYERS': [256, 256, 256],
        'GLOBAL_INFORMATION': {
            'MLP_LAYERS': [256, 256, 256]
        }
    })

    model = GNN(cfg)

    batch_dict = model(batch_dict)
    edges = batch_dict['gnn_edges']
    assert edges.shape[0] == 2
    assert edges.shape[1] == 6
