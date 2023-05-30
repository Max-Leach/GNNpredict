from torch import nn
import torch

from novel_arch.deep_attn.feat_evolve import OrderedGraphFeatUpdate
from novel_arch.deep_attn.state_mlp import ConcatStateMLP

class DeepAtomSum(nn.Module): ### NOTE: we may use a custom aggregator for this class for atom update
    ''' deeper state evolution, just add nearby atoms + edges for atom feat update '''
    def __init__(self):
        super().__init__()

        feat_updaters = {
            'bond' : None,
            'atom' : None,
            'global' : None,
        }
        graph_layer_0 = OrderedGraphFeatUpdate(['bond', 'atom', 'global'], feat_updaters)

    def forward(self, feats, graph):
        ## graph processing network
        feats = graph_layer_0(feats, graph)

        ## reaction graph construction via difference of component graphs

        ## difference of reactant and product features to get reaction graph features

        ## set2set to get 1 feature vector for each node type

        ## concat -> MLP to scalar