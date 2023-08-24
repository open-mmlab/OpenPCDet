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
    def __init__(self, in_dim, out_dim):
        super(EdgeConv, self).__init__(aggr='max')
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        x = torch.cat((x_j - x_i, x_i), dim=-1)
        x = self.fc(x)
        x = F.relu(x)
        return x


class GNN(nn.Module):
    def __init__(self, object_relation_cfg):
        super(GNN, self).__init__()
        self.graph_cfg = object_relation_cfg.GRAPH
        self.gnn_layers = object_relation_cfg.LAYERS
        self.global_information = object_relation_cfg.GLOBAL_INFORMATION if 'GLOBAL_INFORMATION' in object_relation_cfg  else None

        if self.global_information:
            self.global_mlp = self.global_information.MLP_LAYERS
            mlp_layer_list = []
            for i in range(len(self.global_mlp)):
                if i == 0:
                    mlp_layer_list.append(nn.Linear(7, self.global_mlp[i]))
                else:
                    mlp_layer_list.append(nn.Linear(self.global_mlp[i-1], self.global_mlp[i]))
                mlp_layer_list.append(nn.ReLU())
            self.global_info_mlp = nn.Sequential(*mlp_layer_list)
        
        self.gnn_input_dim = (self.global_mlp[-1] if self.global_information else 0) + 256
        conv_layer_list = []
        for i in range(len(self.gnn_layers)):
            if i == 0:
                conv_layer_list.append(EdgeConv(2*self.gnn_input_dim, self.gnn_layers[i]))
            else:
                conv_layer_list.append(EdgeConv(2*self.gnn_layers[i-1], self.gnn_layers[i]))
        self.gnn = nn.ModuleList(conv_layer_list)
        self.init_weights()


    def init_weights(self, weight_init='xavier'):
        if weight_init == 'xavier':
            init_func = nn.init.xavier_uniform_
        for m in self.gnn:
            init_func(m.fc.weight)
        if self.global_information:
            for m in self.global_info_mlp:
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
    
    def forward(self, batch_dict):
        # BxNx7
        proposal_boxes = batch_dict['rois']
        # BxNxC
        pooled_features = batch_dict['pooled_features']
        B,N,C = pooled_features.shape

        if self.global_information:
            embedded_global_features = self.global_info_mlp(proposal_boxes)
            pooled_features = torch.cat((pooled_features, embedded_global_features), dim=-1)

        pooled_features = pooled_features.view(-1, self.gnn_input_dim)

        batch_vector = torch.arange(B, device=pooled_features.device).repeat_interleave(N)
        if self.graph_cfg.NAME == 'radius_graph':
            edge_index = tg.nn.radius_graph(proposal_boxes[:,:,:3].view(-1, 3), r=self.graph_cfg.RADIUS, batch=batch_vector, loop=False)
        elif self.graph_cfg.NAME == 'knn':
            edge_index = tg.nn.knn_graph(proposal_boxes[:,:,:3].view(-1, 3), k=self.graph_cfg.K, batch=batch_vector, loop=False)
            
        batch_dict['gnn_edges'] = edge_index

        gnn_features = [pooled_features]
        x = pooled_features
        for i in range(len(self.gnn)):
            x = self.gnn[i](x, edge_index)
            gnn_features.append(x)

        batch_dict['related_features'] = torch.cat(gnn_features, dim=-1)

        return batch_dict

if __name__ == '__main__':
    rois = torch.tensor([
        [[0, 0, 0], [2, 2, 2], [4, 4, 4]]
    ], dtype=torch.float32)
    pooled_features = torch.tensor([
        [[1, 2], [2, 3], [3, 4]]
    ], dtype=torch.float32)
    batch_dict = {
        'rois': rois,  # Random positions for 10 batches of 100 proposals each
        'pooled_features': pooled_features  # Random 16-dimensional features for 10 batches of 100 proposals each
    }


    model = GNN(2, 4)
    print(model(batch_dict))
