from torch import nn
import torch
from itertools import pairwise

from novel_arch.deep_attn.feat_evolve import OrderedGraphFeatUpdate
from novel_arch.deep_attn.state_mlp import ConcatStateMLP
from novel_arch.deep_attn.feat_type_updaters import EdgeNeighborUpdate, AtomAggregUpdate, GlobalAggregUpdate
from novel_arch.deep_attn.feat_type_updaters import concat_sum_atom_edge_feat, atom_mean, bond_mean

from bondnet.model.gated_reaction_network import mol_graph_to_rxn_graph
from bondnet.layer.readout import Set2SetThenCat

class DeepAtomSum(nn.Module): ### NOTE: we may use a custom aggregator for this class for atom update
    ''' deeper state evolution, just add nearby atoms + edges for atom feat update '''
    def __init__(self, in_feat_sizes, embedding_size, graph_hidden_size, graph_layers, residual=True):
        super().__init__()

        self.embedders = {k : nn.Linear(in_feat_sizes[k], embedding_size) for k in in_feat_sizes}
        embed_feat_sizes = {'bond': embedding_size, 'atom': embedding_size, 'global': embedding_size}
        feat_sizes = [embed_feat_sizes] + graph_layers * [graph_hidden_size]

        graph_net = []
        for in_s, out_s in pairwise(feat_sizes): # this for loop is to deal with case that hidden sizes are different, otherwise its more complicated than needed
            in_feat_sizes = in_s
            if not isinstance(in_feat_sizes, dict):
                in_feat_sizes = {'bond': in_s, 'atom': in_s, 'global': in_s} # unified size for rest of graph layers
            bond_updt = EdgeNeighborUpdate(in_feat_sizes, out_s, residual=residual)
            in_feat_sizes['bond'] = out_s
            atom_updt = AtomAggregUpdate(concat_sum_atom_edge_feat, in_feat_sizes, out_s, residual=residual)
            in_feat_sizes['atom'] = out_s
            global_updt = GlobalAggregUpdate(bond_mean(), atom_mean(), in_feat_sizes, out_s, residual=residual)

            feat_updaters = {
                'bond' : bond_updt,
                'atom' : atom_updt,
                'global' : global_updt,
            }
            graff_layer = OrderedGraphFeatUpdate(['bond', 'atom', 'global'], feat_updaters)
            graph_net.append(graff_layer)
        self.graph_net = graph_net

        ntypes = ["atom", "bond"]
        self.readout = Set2SetThenCat( # from bondnet!!
            n_iters=6, n_layer=3, ntypes=ntypes, in_feats=[graph_hidden_size] * len(ntypes), ntypes_direct_cat=["global"]
        )

        self.fc_to_scalar = nn.ModuleList()
        in_size = graph_hidden_size * 2 + graph_hidden_size * 2 + graph_hidden_size
        for s in [128, 64]:
            out_size = s
            self.fc_to_scalar.append(nn.Linear(in_size, out_size))
            self.fc_to_scalar.append(nn.BatchNorm1d(out_size))
            self.fc_to_scalar.append(nn.ReLU()) #batchnorm + droptout before or afer this point pls

            in_size = out_size
        self.fc_to_scalar.append(nn.Linear(in_size, 1))

    def forward(self, graph, feats, reactions):
        g = graph.local_var()
        ## graph processing network
        #load feats to graph
        # for fname in feats:
        #     g.nodes[fname].data.update({'ft': feats[fname]})
        # feats = self.graph_net(feats, g)
        for ftype in self.embedders: # unify all sizes
            feats[ftype] = self.embedders[ftype](feats[ftype])
        for gl in self.graph_net:
            feats = gl(feats, graph)

        ## reaction graph construction via difference of component graphs
        ## difference of reactant and product features to get reaction graph features
        graph, feats = mol_graph_to_rxn_graph(graph, feats, reactions) # from bondnet!!

        ## set2set to get 1 feature vector for each node type
        # we'll separate these later
        ## concat -> MLP to scalar
        feats = self.readout(graph, feats)

        for fc in self.fc_to_scalar:
            feats = fc(feats)

        return feats