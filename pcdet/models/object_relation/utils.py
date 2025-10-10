import torch.nn as nn

def build_mlp(input_dim, hidden_dims, activation='ReLU', bn=False, drop_out=None):
    mlp_list = []
    for i in range(len(hidden_dims)):
        if i == 0:
            mlp_list.append(nn.Linear(input_dim, hidden_dims[i]))
        else:
            mlp_list.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        if bn:
            mlp_list.append(nn.BatchNorm1d(hidden_dims[i]))
        mlp_list.append(getattr(nn, activation)())
        if drop_out:
            mlp_list.append(nn.Dropout(drop_out))
    return nn.Sequential(*mlp_list)
