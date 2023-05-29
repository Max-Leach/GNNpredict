from torch import nn
import torch
from itertools import chain, pairwise

class ConcatStateMLP(nn.Module):
    ''' concat dict of features in same order for passes through an instance of this,
        pass thru MLP '''
    def __init__(self, feat_sizes, out_size, inner_layer_sizes=[], bias=True):
        # feat_sizes - dict of feature names -> its size
        # out_size - output vector size
        # inner_layer_sizes - list of fc layer sizes between in and out, empty list means this will be a single layer MLP
        super().__init__()

        self.feat_order = feat_sizes.keys()
        first_size = sum(feat_sizes.values())
        layer_sizes = [first_size] + inner_layer_sizes + [out_size]
        fc_nested = [(nn.Linear(fc_lens[0], fc_lens[1], bias=bias), nn.ReLU()) for fc_lens in pairwise(layer_sizes)]
        self.fc_layers = nn.Sequential(*tuple(chain(*fc_nested)))

    def forward(self, feats):
        x = torch.cat([feats[nt] for nt in self.feat_order], dim=-1)
        return self.fc_layers(x)