from torch import nn
import torch
from itertools import chain, pairwise

''' given out, in, and inner sizes of layers, produce activated fc layers - NOTE: batch norm is false by default!
'''
def mlp_from_sizes(in_size, out_size, inner_layer_sizes=[], bias=True, batch_norm=False):
    layer_sizes = [in_size] + inner_layer_sizes + [out_size]
    fc_nested = [fc_layer(fc_lens[0], fc_lens[1], batch_norm, bias) for fc_lens in pairwise(layer_sizes)]
    return nn.Sequential(*tuple(chain(*fc_nested)))

def fc_layer(in_size, out_size, batch_norm, bias):
    lay = [nn.Linear(in_size, out_size, bias=bias)]
    if batch_norm:
        lay.append(nn.BatchNorm1d(out_size))
    return lay + [nn.ReLU()]

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
        self.fc_layers = mlp_from_sizes(first_size, out_size, inner_layer_sizes)

    def forward(self, feats):
        x = torch.cat([feats[nt] for nt in self.feat_order], dim=-1)
        return self.fc_layers(x)