from torch import nn
import torch
from itertools import pairwise

from novel_arch.deep_attn.feat_evolve import OrderedGraphFeatUpdate
from novel_arch.deep_attn.feat_type_updaters import EdgeNeighborUpdate, AtomAggregUpdate, GlobalAggregUpdate
from novel_arch.deep_attn.feat_type_updaters import atom_mean, bond_mean
from novel_arch.deep_attn.readout import Set2Set
from novel_arch.deep_attn.rxn_graph import bondnet_batch_to_own

class DeepAtom(nn.Module):
    ''' deeper state evolution, just add nearby atoms + edges for atom feat update '''
    ''' graph_inner_layer_sizes - how wide individual layers in gnn portion will be, is independent of graph_layers count '''
    def __init__(self, atom_aggregators, b2g_aggregator, a2g_aggregator, in_feat_sizes, graph_hidden_size, graph_layers, graph_inner_layer_sizes=[], residual=True, fc_readout_sizes=[128, 64], set2set_iters=6, set2set_layers=3, atom_include_edges=True):
        super().__init__()

        embedding_size = graph_hidden_size
        self.embedders = nn.ModuleDict({k : nn.Linear(in_feat_sizes[k], embedding_size) for k in in_feat_sizes})
        self.assemble_gnn(
            in_feat_sizes, embedding_size, graph_hidden_size, graph_layers, graph_inner_layer_sizes, residual, atom_aggregators, atom_include_edges,
            b2g_aggregator, a2g_aggregator,
            )

        self.set2set_extract = nn.ModuleDict({
            'atom': Set2Set(graph_hidden_size, set2set_iters, set2set_layers, 'atom'),
            'bond': Set2Set(graph_hidden_size, set2set_iters, set2set_layers, 'bond'),
        })
        self.direct_concat = ['global']
        self.concat_order = ['bond', 'atom', 'global']

        self.fc_to_scalar = nn.ModuleList()
        in_size = graph_hidden_size * 2 + graph_hidden_size * 2 + graph_hidden_size
        for s in fc_readout_sizes:
            out_size = s
            self.fc_to_scalar.append(nn.Linear(in_size, out_size))
            self.fc_to_scalar.append(nn.BatchNorm1d(out_size))
            self.fc_to_scalar.append(nn.ReLU()) #batchnorm + droptout before or afer this point pls

            in_size = out_size
        self.fc_to_scalar.append(nn.Linear(in_size, 1))
    
    def assemble_gnn(self, in_feat_sizes, embedding_size, graph_hidden_size, graph_layers, graph_inner_layer_sizes, residual, atom_aggregators, atom_include_edges, b2g_aggregator, a2g_aggregator):
        embed_feat_sizes = {'bond': embedding_size, 'atom': embedding_size, 'global': embedding_size}
        feat_sizes = [embed_feat_sizes] + graph_layers * [graph_hidden_size]

        graph_net = nn.ModuleList()
        for i, (in_s, out_s) in enumerate(pairwise(feat_sizes)): # this for loop is to deal with case that hidden sizes are different, otherwise its more complicated than needed
            in_feat_sizes = in_s
            if not isinstance(in_feat_sizes, dict):
                in_feat_sizes = {'bond': in_s, 'atom': in_s, 'global': in_s} # unified size for rest of graph layers
            inner_layer_sizes = []
            if i < len(graph_inner_layer_sizes):
                inner_layer_sizes = graph_inner_layer_sizes[i]
            try:
                atom_aggregator = atom_aggregators[i]
            except TypeError:
                atom_aggregator = atom_aggregators
            bond_updt = EdgeNeighborUpdate(in_feat_sizes, out_s, inner_layer_sizes=inner_layer_sizes, residual=residual)
            in_feat_sizes['bond'] = out_s
            atom_updt = AtomAggregUpdate(atom_aggregator, in_feat_sizes, out_s, inner_layer_sizes=inner_layer_sizes, residual=residual, include_edges=atom_include_edges)
            in_feat_sizes['atom'] = out_s
            global_updt = GlobalAggregUpdate(b2g_aggregator, a2g_aggregator, in_feat_sizes, out_s, inner_layer_sizes=inner_layer_sizes, residual=residual)

            feat_updaters = nn.ModuleDict({
                'bond' : bond_updt,
                'atom' : atom_updt,
                'global' : global_updt,
            })
            graff_layer = OrderedGraphFeatUpdate(['bond', 'atom', 'global'], feat_updaters)
            graph_net.append(graff_layer)
        self.graph_net = graph_net
    
    def forward(self, graph, feats, reactions):
        ''' 
            graph: batched dgl graph
            feats: dict of torch tensor feats for batched graph
            reactions: encoding method of reaction, references individual mol graphs in graph batch
        '''
        g = graph.local_var()
        ## graph processing network
        #load feats to graph
        for ftype in self.embedders: # unify all sizes
            feats[ftype] = self.embedders[ftype](feats[ftype])
        for gl in self.graph_net:
            feats = gl(feats, graph)

        ## reaction graph construction via difference of component graphs
        ## difference of reactant and product features to get reaction graph features
        # graph, feats = mol_graph_to_rxn_graph(graph, feats, reactions) # from bondnet!!
        ''' reaction graph construction via own method '''
        feats, graph = bondnet_batch_to_own(graph, feats, reactions)

        ## set2set to get 1 feature vector for each node type
        encoded_feats = {nt : self.set2set_extract[nt](graph, feats[nt]) for nt in self.set2set_extract}
        direct_feats = {nt : feats[nt] for nt in self.direct_concat}
        encoded_feats.update(direct_feats)
        ## concat -> MLP to scalar
        feats = torch.cat([encoded_feats[nt] for nt in self.concat_order], dim=-1)

        # feats = self.readout(graph, feats)

        for fc in self.fc_to_scalar:
            feats = fc(feats)

        return feats