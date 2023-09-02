import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
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
    def __init__(self, in_dim, out_dim, drop_out=None):
        super(EdgeConv, self).__init__(aggr='max')
        # self.fc = nn.Linear(in_dim, out_dim)
        fc_list = []
        fc_list.append(nn.Linear(in_dim, out_dim))
        # also try graph norm
        # fc_list.append(tg.nn.GraphNorm(out_dim))
        fc_list.append(tg.nn.BatchNorm(out_dim))
        fc_list.append(nn.ReLU())
        if drop_out:
            fc_list.append(nn.Dropout(drop_out))
        
        self.fc = nn.Sequential(*fc_list)

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr is not None:
            x = torch.cat((x_j - x_i, x_i, edge_attr), dim=-1)
        else:
            x = torch.cat((x_j - x_i, x_i), dim=-1)
        x = self.fc(x)
        return x


class GNN(nn.Module):
    def __init__(self, object_relation_cfg, number_classes=3, pooled_feature_dim=256):
        super(GNN, self).__init__()
        self.graph_cfg = object_relation_cfg.GRAPH
        self.gnn_layers = object_relation_cfg.LAYERS
        self.global_information = object_relation_cfg.GLOBAL_INFORMATION if 'GLOBAL_INFORMATION' in object_relation_cfg  else None
        self.number_classes = number_classes
        self.drop_out = object_relation_cfg.DP_RATIO
        self.pooled_feature_dim = pooled_feature_dim

        if self.global_information:
            self.global_mlp = self.global_information.MLP_LAYERS
            mlp_layer_list = []
            global_mlp_input_dim = pooled_feature_dim + 7 if self.global_information.CONCATENATED else 7
            for i in range(len(self.global_mlp)):
                if i == 0:
                    mlp_layer_list.append(nn.Linear(global_mlp_input_dim, self.global_mlp[i]))
                else:
                    mlp_layer_list.append(nn.Linear(self.global_mlp[i-1], self.global_mlp[i]))
                
                mlp_layer_list.append(nn.BatchNorm1d(self.global_mlp[i]))
                mlp_layer_list.append(nn.ReLU())
                if self.drop_out:
                    mlp_layer_list.append(nn.Dropout(self.drop_out))

            self.global_info_mlp = nn.Sequential(*mlp_layer_list)
        
        gnn_input_dim = self.global_mlp[-1] if self.global_information.CONCATENATED else (self.global_mlp[-1] + self.pooled_feature_dim)
        conv_layer_list = []
        for i in range(len(self.gnn_layers)):
            if i == 0:
                input_dim = 2*gnn_input_dim + (7 if self.graph_cfg.EDGE_EMBEDDING else 0)
            else:
                input_dim = 2*self.gnn_layers[i-1]
            conv_layer_list.append(EdgeConv(input_dim, self.gnn_layers[i], drop_out=self.drop_out))
        self.gnn = nn.ModuleList(conv_layer_list)
        self.init_weights()


    def init_weights(self, weight_init='xavier'):
        if weight_init == 'xavier':
            init_func = nn.init.xavier_uniform_
        for edge_convs in self.gnn:
            for m in edge_convs.fc:
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
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

        edge_index = self.get_edges(proposal_boxes[:,:3], proposal_labels, (B, N, C))
        batch_dict['gnn_edges'] = edge_index

        edge_attr = None
        if self.graph_cfg.EDGE_EMBEDDING:
            from_node, to_node = edge_index
            edge_attr = proposal_boxes[from_node] - proposal_boxes[to_node]
        
        gnn_features = [pooled_features]
        x = pooled_features
        for i in range(len(self.gnn)):
            x = self.gnn[i](x, edge_index, edge_attr=edge_attr)
            gnn_features.append(x)

        batch_dict['related_features'] = torch.cat(gnn_features, dim=-1)

        return batch_dict

    def get_edges(self, proposal_boxes, proposal_labels, shape):
        B, N, _ = shape
        f = getattr(tg.nn, self.graph_cfg.NAME)
        a = (self.graph_cfg.RADIUS if self.graph_cfg.NAME == 'radius_graph' else self.graph_cfg.K)
        batch_vector = torch.arange(B, device=proposal_boxes.device).repeat_interleave(N)
        
        if self.graph_cfg.CONNECT_ONLY_SAME_CLASS:
            final_edge_indices = []
            for predicted_class in range(1, self.number_classes + 1):
                label_indices = torch.where(proposal_labels == predicted_class)[0]
                label_proposal_boxes = proposal_boxes[label_indices]
                label_batch_vector = batch_vector[label_indices]
                
                label_edge_index = f(label_proposal_boxes, a, batch=label_batch_vector, loop=False)
                label_edge_index[0] = label_indices[label_edge_index[0]]
                label_edge_index[1] = label_indices[label_edge_index[1]]
                final_edge_indices.append(label_edge_index)
            edge_index = torch.cat(final_edge_indices, dim=-1)
        else:
            edge_index = f(proposal_boxes, a, batch=batch_vector, loop=False)
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
