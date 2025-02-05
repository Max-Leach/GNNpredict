from torch import nn
import torch
from itertools import pairwise

from architecture.feat_evolve import OrderedGraphFeatUpdate
from architecture.feat_type_updaters import EdgeNeighborUpdate, AtomAggregUpdate, GlobalAggregUpdate
from architecture.data.rxn_graph import get_rxn_feat_list

class DeepBDE(nn.Module):
    ''' deeper state evolution, just add nearby atoms + edges for atom feat update '''
    ''' graph_inner_layer_sizes - how wide individual layers in gnn portion will be, is independent of graph_layers count '''
    def __init__(self, atom_aggregators, b2g_aggregators, a2g_aggregators, in_feat_sizes, graph_hidden_size, graph_layers, graph_inner_layer_sizes=[], residual=True, fc_readout_sizes=[128, 64], dropout=0.0, atom_include_edges=True, activation_fn=None, readout_relu=False):
        super().__init__()

        if activation_fn == None:
            activation_fn = nn.ReLU

        embedding_size = graph_hidden_size
        self.embedders = nn.ModuleDict({k : nn.Linear(in_feat_sizes[k], embedding_size) for k in in_feat_sizes})
        self.assemble_gnn(
            in_feat_sizes, embedding_size, graph_hidden_size, graph_layers, graph_inner_layer_sizes, residual, atom_aggregators, atom_include_edges,
            b2g_aggregators, a2g_aggregators, dropout=dropout, activation_fn=activation_fn
            )
        
        self.direct_concat = ['global']
        self.concat_order = ['bond', 'atom', 'global']

        self.fc_to_scalar = nn.ModuleList()
        in_size = len(graph_inner_layer_sizes) * (graph_hidden_size * 3)
        
        for s in fc_readout_sizes:
            out_size = s
            self.fc_to_scalar.append(nn.Linear(in_size, out_size))
            self.fc_to_scalar.append(nn.BatchNorm1d(out_size))
            self.fc_to_scalar.append(nn.Dropout1d(p=0.0))
            
            if readout_relu:
                self.fc_to_scalar.append(nn.ReLU())
            else:
                self.fc_to_scalar.append(activation_fn())

            in_size = out_size
        self.fc_to_scalar.append(nn.Linear(in_size, 1))
    
    def assemble_gnn(self, in_feat_sizes, embedding_size, graph_hidden_size, graph_layers, graph_inner_layer_sizes, residual, atom_aggregators, atom_include_edges, b2g_aggregators, a2g_aggregators, dropout, activation_fn):
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
            bond_updt = EdgeNeighborUpdate(in_feat_sizes, out_s, inner_layer_sizes=inner_layer_sizes, residual=residual, dropout=dropout, activation_fn=activation_fn)
            in_feat_sizes['bond'] = out_s
            atom_updt = AtomAggregUpdate(atom_aggregator, in_feat_sizes, out_s, inner_layer_sizes=inner_layer_sizes, residual=residual, include_edges=atom_include_edges, dropout=dropout, activation_fn=activation_fn)
            in_feat_sizes['atom'] = out_s
            try:
                b2g_aggregator = b2g_aggregators[i]
            except TypeError:
                b2g_aggregator = b2g_aggregators
            try:
                a2g_aggregator = a2g_aggregators[i]
            except TypeError:
                a2g_aggregator = a2g_aggregators
            global_updt = GlobalAggregUpdate(b2g_aggregator, a2g_aggregator, in_feat_sizes, out_s, inner_layer_sizes=inner_layer_sizes, residual=residual, dropout=dropout, activation_fn=activation_fn)

            feat_updaters = nn.ModuleDict({
                'bond' : bond_updt,
                'atom' : atom_updt,
                'global' : global_updt,
            })
            graff_layer = OrderedGraphFeatUpdate(['bond', 'atom', 'global'], feat_updaters)
            graph_net.append(graff_layer)
        self.graph_net = graph_net
    
    def forward(self, graph, feats, rxns):
        ''' 
            graph: batched dgl graph
            feats: dict of torch tensor feats for batched graph
            reactions: encoding method of reaction, references individual mol graphs in graph batch
        '''
        g = graph.local_var()
        ## graph processing network
        # load feats to graph
        for ftype in self.embedders: # unify all sizes
            feats[ftype] = self.embedders[ftype](feats[ftype])
        graph_feats = [feats]
        for gl in self.graph_net:
            # feats = gl(feats, graph)
            graph_feats.append(gl(graph_feats[-1], g))
        graph_feats = graph_feats[1:]

        ## reaction graph construction via difference of component graphs
        ## difference of reactant and product features to get reaction graph features
        ''' reaction graph construction via own method '''
        feats = {nt : torch.cat([gf[nt] for gf in graph_feats], dim=-1) for nt in self.concat_order}
        indiv_feats = get_rxn_feat_list(g, feats, rxns)

        # sum and concat across all graph layers
        encoded_feats = [{fn : torch.sum(ftd[fn], dim=0) for fn in ftd} for ftd in indiv_feats]
        encoded_feats = {nt : torch.stack([f[nt] for f in encoded_feats], dim=0) for nt in self.concat_order}

        ## concat -> MLP to scalar
        feats = torch.cat([encoded_feats[nt] for nt in self.concat_order], dim=-1)

        for fc in self.fc_to_scalar:
            feats = fc(feats)

        return feats