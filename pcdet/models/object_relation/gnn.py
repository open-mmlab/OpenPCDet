import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
import torch_scatter

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
    def __init__(self, dim, radius, layers):
        super(GNN, self).__init__()
        self.radius = radius
        conv_list = []
        for i in range(len(layers)):
            if i == 0:
                conv_list.append(EdgeConv(2*(256+7), layers[i]))
            else:
                conv_list.append(EdgeConv(2*layers[i-1], layers[i]))
        self.conv_list = nn.ModuleList(conv_list)
        self.radius_transform = tg.transforms.RadiusGraph(r=self.radius)
        self.init_weights()

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'xavier':
            init_func = nn.init.xavier_uniform_
        for m in self.conv_list:
            init_func(m.fc.weight)
    
    def forward(self, batch_dict):
        # BxNx7
        proposal_boxes = batch_dict['rois']
        # BxNxC
        pooled_features = batch_dict['pooled_features']
        B,N,C = pooled_features.shape

        global_pooled_features = torch.cat((pooled_features, proposal_boxes), dim=-1)
        global_pooled_features = global_pooled_features.view(-1, C+7)

        batch_vector = torch.arange(B, device=pooled_features.device).repeat_interleave(N)
        edge_index = tg.nn.radius_graph(proposal_boxes[:,:,:3].view(-1, 3), r=self.radius, batch=batch_vector, loop=False)
        batch_dict['gnn_edges'] = edge_index

        gnn_features = [global_pooled_features]
        x = global_pooled_features
        for i in range(len(self.conv_list)):
            x = self.conv_list[i](x, edge_index)
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
