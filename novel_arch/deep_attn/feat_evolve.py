from torch import nn
import torch

class OrderedGraphFeatUpdate(nn.Module):
    '''
        Update feature types in graph in certain order and with provided operations
    '''
    def __init__(self, feat_update_order, feat_updaters):
        # feat_update_order - list of feature names in update order
        # feat_updaters - dict of feature name -> update fn
        super().__init__()

        self.update_order = feat_update_order
        self.updaters = feat_update_order
    
    def forward(self, feats):
        for fname in self.update_order:
            feats[fname] = self.updaters[fname](feats)
        return feats