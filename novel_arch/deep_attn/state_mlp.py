from torch import nn
import torch
from itertools import chain, pairwise

''' given out, in, and inner sizes of layers, produce activated fc layers - NOTE: batch norm is false by default!
'''
def mlp_from_sizes(in_size, out_size, inner_layer_sizes=[], bias=True, batch_norm=False, dropout=False):
    layer_sizes = [in_size] + inner_layer_sizes + [out_size]
    fc_nested = [fc_layer(fc_lens[0], fc_lens[1], batch_norm, dropout, bias) for fc_lens in pairwise(layer_sizes)]
    return nn.Sequential(*tuple(chain(*fc_nested)))

def fc_layer(in_size, out_size, batch_norm, dropout, bias):
    lay = [nn.Linear(in_size, out_size, bias=bias)]
    if batch_norm:
        lay.append(nn.BatchNorm1d(out_size))
    if dropout:
        lay.append(nn.Dropout1d(p=0.0))
    return lay + [nn.ReLU()]