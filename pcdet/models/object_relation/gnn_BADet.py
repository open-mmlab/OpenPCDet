from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU
import torch
import torch.nn as nn
import torch_geometric as tg
from torch_geometric.data import Data
from torch_geometric.transforms import RadiusGraph

class StateConvLayer(MessagePassing):
    def __init__(self, state_dim):
        """
        Graph Layer to perform update of the node states.
        """
        super(StateConvLayer, self).__init__(aggr='max')

        # MLP to transform node state into the relative offset to alleviate translation variance.
        self.mlp_h = Seq(Linear(state_dim, state_dim//2), 
                         ReLU(inplace=True),
                         Linear(state_dim//2, state_dim//4), 
                         ReLU(inplace=True),
                         Linear(state_dim//4, 3))

        # MLP to compute edge features
        self.mlp_f = Seq(Linear(state_dim+3, state_dim//2),
                         ReLU(inplace=True), 
                         Linear(state_dim//2, state_dim//4), 
                         ReLU(inplace=True),
                         Linear(state_dim//4, state_dim),
                         )

        self.mlp_g = Seq(Linear(state_dim, state_dim//2),
                         ReLU(inplace=True), 
                         Linear(state_dim//2, state_dim//4), 
                         ReLU(inplace=True),
                         Linear(state_dim//4,state_dim),
                         )

    def forward(self, s, x, edge_index):
        return self.propagate(edge_index, s=s, x=x)

    def message(self, x_j, x_i, s_i, s_j):

        # The extended graph update algorithm.
        delta_x_i = self.mlp_h(s_i)
        tmp = torch.cat([x_j - x_i - delta_x_i, s_j], dim=1)
        e_ij = self.mlp_f(tmp)
        return e_ij

    def update(self, e_ij, s):
        # Update vertex state based on aggregated edge features
        return s + self.mlp_g(e_ij)

def basic_block(in_channel, out_channel):
    """
    Create block with linear layer followed by IN and ReLU.
    :param in_channel: number of input features
    :param out_channel: number of output features
    :return: PyTorch Sequential object
    """
    return nn.Sequential(Linear(in_channel, out_channel),
                         nn.InstanceNorm1d(out_channel),
                         nn.ReLU(inplace=True))

class BARefiner(nn.Module):
    def __init__(self, object_relation_cfg, number_classes=3):
        """
        Boundary-Aware Graph Neural Network, which takes 3D proposals in immediate neighborhood
        as inputs for graph construction within a given cut-off distance, associating 3D proposals 
        in the form of local neighborhood graph, with boundary correlations of an object being 
        explicitly informed through an information compensation mechanism.

        Args:
        :param state_dim: maximum number of state features
        :param n_classes: number of classes
        :param n_iterations: number of GNN iterations to perform
        """
        super(BARefiner, self).__init__()
        state_dim = object_relation_cfg.STATE_DIM
        n_iterations = object_relation_cfg.ITERATIONS
        self.graph_cfg = object_relation_cfg.GRAPH
        self.n_classes = 1
        self._num_anchor_per_loc = 1
        self._box_code_size = 7

        # List of GNN layers
        self.graph_layers = nn.ModuleList([StateConvLayer(state_dim) for _ in
                                     range(n_iterations)])
        
        # MLP for class prediction
        self.mlp_class = Seq(basic_block(state_dim, state_dim),
                             basic_block(state_dim, state_dim),
                             Linear(state_dim, self._num_anchor_per_loc * self.n_classes))

        # Set of MLPs for per-class bounding box regression
        self.mlp_loc = Seq(basic_block(state_dim, state_dim),
                                          basic_block(state_dim, state_dim),
                                          Linear(state_dim, self._num_anchor_per_loc * self._box_code_size))

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


    def forward(self, batch_dict):
        (B, N, C) = batch_dict['pooled_features'].shape

        # BxNx7
        proposal_boxes = batch_dict['rois'].view(B*N,7)
        # BxN
        proposal_labels = batch_dict['roi_labels'].view(B*N)
        # BxNxC
        pooled_features = batch_dict['pooled_features'].view(B*N,C)

        edge_index = self.get_edges(proposal_boxes[:,:3], proposal_labels, (B, N, C))

        
        # Set initial vertex state
        # state = batch_data['node_features']
        # Perform GNN computations
        for graph_layer in self.graph_layers:
            # Update vertex state
            x = graph_layer(pooled_features, proposal_boxes[:,:3], edge_index)

        x = x.unsqueeze(0)
        cls_pred = self.mlp_class(x)
        reg_pred = self.mlp_loc(x)
        batch_dict['rcnn_cls'] = cls_pred.view(B*N, self.n_classes)
        batch_dict['rcnn_reg'] = reg_pred.view(B*N, self._box_code_size)
        return batch_dict
        