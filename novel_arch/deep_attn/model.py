from torch import nn
import torch

from novel_arch.deep_attn.feat_evolve import OrderedGraphFeatUpdate
from novel_arch.deep_attn.state_mlp import ConcatStateMLP
from novel_arch.deep_attn.feat_type_updaters import EdgeNeighborUpdate, AtomAggregUpdate, GlobalAggregUpdate
from novel_arch.deep_attn.feat_type_updaters import concat_sum_atom_edge_feat, atom_mean, bond_mean

class DeepAtomSum(nn.Module): ### NOTE: we may use a custom aggregator for this class for atom update
    ''' deeper state evolution, just add nearby atoms + edges for atom feat update '''
    def __init__(self, in_feat_sizes):
        super().__init__()

        feat_updaters = {
            'bond' : EdgeNeighborUpdate(in_feat_sizes),
            'atom' : AtomAggregUpdate(concat_sum_atom_edge_feat, in_feat_sizes),
            'global' : GlobalAggregUpdate(bond_mean(), atom_mean(), in_feat_sizes),
        }
        self.graph_layer_0 = OrderedGraphFeatUpdate(['bond', 'atom', 'global'], feat_updaters)

    def forward(self, feats, graph):
        g = graph.local_var()
        ## graph processing network
        #load feats to graph
        for fname in feats:
            g.nodes[fname].data.update({'ft': feats[fname]})
        feats = self.graph_layer_0(feats, g)
        return feats

        ## reaction graph construction via difference of component graphs

        ## difference of reactant and product features to get reaction graph features

        ## set2set to get 1 feature vector for each node type

        ## concat -> MLP to scalar