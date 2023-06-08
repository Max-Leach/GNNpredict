from torch import nn
import torch
from dgl import function as fn
import dgl
from itertools import pairwise, chain
from novel_arch.deep_attn.state_mlp import mlp_from_sizes

class EdgeNeighborUpdate(nn.Module):
    ''' 
        MLP update of bond, atom, global features
    '''
    def __init__(self, in_feat_sizes, out_size, inner_layer_sizes=[], residual=False, bias=False):
        super().__init__()

        self.residual = residual
        in_mlp_size = sum(in_feat_sizes.values())
        self.fc = mlp_from_sizes(in_mlp_size, out_size, inner_layer_sizes, bias=bias, batch_norm=True)

    def forward(self, feats, graph): # features are assumed to be loaded in before this fn
        g = graph.local_var()

        g.multi_update_all(
            {
                'a2b' : (fn.copy_u('ft', 'm'), fn.sum('m', 'a_sum')),
                'g2b' : (fn.copy_u('ft', 'm'), fn.sum('m', 'g'))
            },
            'sum'
        )
        b = feats['bond']
        a_sum = g.nodes['bond'].data['a_sum']
        glob = g.nodes['bond'].data['g']
        fc_in = torch.cat([b, a_sum, glob], dim=-1)

        b = self.fc(fc_in)
        if self.residual:
            b = b + feats['bond']
        feats['bond'] = b
        return feats

class AtomAggregUpdate(nn.Module):
    '''
        custom aggregation of node features (especially bonds)
    '''
    def __init__(self, atom_edge_feat_aggreg, in_feat_sizes, out_size, inner_layer_sizes=[], residual=False, bias=False, include_edges=True):
        super().__init__()

        self.atom_edge_feat_aggreg = atom_edge_feat_aggreg
        self.residual = residual
        in_mlp_size = 2*in_feat_sizes['atom'] + in_feat_sizes['global']
        if include_edges:
            in_mlp_size += in_feat_sizes['bond']
        self.fc = mlp_from_sizes(in_mlp_size, out_size, inner_layer_sizes, bias=bias, batch_norm=True)
        self.include_edges = include_edges

    def forward(self, feats, graph):
        g = graph.local_var()

        # store atom indices
        g.nodes['atom'].data.update({'i' : torch.arange(g.num_nodes('atom')).float()})
        # atom features to bond transfer for edge + atom agggregation
        g.update_all(fn.copy_u("ft", "ft_a"), copy_mailbox_feat_repeat_if_single('ft_a'), etype="a2b")
        g.update_all(fn.copy_u("i", "i_a"), copy_mailbox_feat_repeat_if_single('i_a'), etype="a2b")

        g.update_all(copy_multiple_u(['ft_a', 'i_a', 'ft']), self.atom_edge_feat_aggreg, etype='b2a') # this is where we'd be changing the aggregation for atoms
        g.update_all(fn.copy_u('ft', 'm'), fn.sum('m', 'g'), etype='g2a')

        # finish graph processing via concatenation and mlp
        a_b_aggreg = g.nodes['atom'].data['a_b_aggreg']
        fc_in = torch.cat([feats['atom'], a_b_aggreg, g.nodes['atom'].data['g']], dim=-1)
        a = self.fc(fc_in)
        if self.residual:
            a = a + feats['atom']
        feats['atom'] = a

        return feats

def concat_sum_atom_edge_feat(nodes):
    return aggreg_atom_edge_no_repeat(nodes, lambda a,b,nodes: torch.sum(torch.cat([a, b], dim=-1), dim=1))

class AtomEdgeReducer(nn.Module):
    def __init__(self, aggregator):
        super().__init__()
        self.aggregator = aggregator

    def forward(self, nodes):
        return aggreg_atom_edge_no_repeat(nodes, self.aggregator)

''' 
    aggregate incoming atom and bond features for atoms via attention

    atm, it uses gatv2 style attention
'''
class AttnAtomEdgeAggreg(nn.Module):
    def __init__(self, feat_size, internal_attn_size, include_edges=True):
        super().__init__()

        self.activ_in_map = nn.Linear(2*feat_size + feat_size, internal_attn_size)
        self.activ = nn.LeakyReLU()
        self.attn_scalar_map = nn.Linear(internal_attn_size, 1)
        self.softmax = nn.Softmax(dim=-2)
        self.include_edges = include_edges

    def forward(self, incoming_atom_fts, incoming_bond_fts, nodes):
        self_feats_per_incoming = nodes.data['ft'].unsqueeze(1).repeat_interleave(incoming_atom_fts.size(1), dim=1)
        
        combined_attn_in = torch.cat([self_feats_per_incoming, incoming_atom_fts, incoming_bond_fts], dim=-1)
        activ_ins = self.activ(self.activ_in_map(combined_attn_in))
        pre_attn = self.attn_scalar_map(activ_ins)
        attn_weights = self.softmax(pre_attn)

        incoming = incoming_atom_fts
        if self.include_edges:
            incoming = torch.cat([incoming_atom_fts, incoming_bond_fts], dim=-1)
        weighted = torch.mul(attn_weights, incoming)
        aggregated = torch.sum(weighted, dim=1)

        return aggregated

# aggregate incoming atom feat concat with bond feat that connects them
# assumes connected atom features, atom indices, bond features already in bond mailbox
def aggreg_atom_edge_no_repeat(nodes, aggreg):
    ft_a = nodes.mailbox['ft_a']
    i_a = nodes.mailbox['i_a'] # order of idx of atoms should correspond with atom feats of ft_a
    bond_ft = nodes.mailbox['ft']

    # index selection
    indices = torch.where(i_a[:, :, 0] != nodes.nodes().unsqueeze(-1), 0, 1) # match passed indices with atom index that is different from self
    non_self_ft_a = ft_a.gather(2, indices.unsqueeze(-1).unsqueeze(-1).repeat_interleave(ft_a.size(3), dim=3)) # use above indices to select proper features
    non_self_ft_a = non_self_ft_a.squeeze(2)

    # user inputted method of aggregating non-self included edge-atom
    aggregated = aggreg(non_self_ft_a, bond_ft, nodes)

    return {'a_b_aggreg' : aggregated}

class GlobalAggregUpdate(nn.Module):
    def __init__(self, edge_aggreg, atom_aggreg, in_feat_sizes, out_size, inner_layer_sizes=[], residual=False, bias=False):
        super().__init__()

        self.atom_aggreg = atom_aggreg
        self.edge_aggreg = edge_aggreg
        self.residual = residual
        in_mlp_size = sum(in_feat_sizes.values())
        self.fc = mlp_from_sizes(in_mlp_size, out_size, inner_layer_sizes, bias=bias, batch_norm=True)

    def forward(self, feats, graph):
        g = graph.local_var()

        g.update_all(fn.copy_u('ft', 'm'), self.atom_aggreg, etype='a2g')
        g.update_all(fn.copy_u('ft', 'm'), self.edge_aggreg, etype='b2g')

        a = g.nodes['global'].data['a']
        b = g.nodes['global'].data['b']
        fc_in = torch.cat([feats['global'], b, a], dim=-1)
        glob = self.fc(fc_in)
        if self.residual:
            glob = glob + feats['global']
        feats['global'] = glob

        return feats

# aggregators for global feat update
def atom_mean():
    return fn.mean('m', 'a')
def bond_mean():
    return fn.mean('m', 'b')

# return reducer fn to just plant full copy of mailbox as feat
def copy_mailbox_feat(feat_name):
    return lambda nodes: {feat_name : nodes.mailbox[feat_name]}

# for single atom case, phantom bond exists
def copy_mailbox_feat_repeat_if_single(feat_name):
    return lambda nodes: copy_mailbox_feat_fn_repeat(nodes, feat_name)
    
def copy_mailbox_feat_fn_repeat(nodes, feat_name):
    m = nodes.mailbox[feat_name]
    if m.shape[1] == 1:
        m = m.repeat_interleave(2, dim=1)
    return {feat_name : m}

# propagate multiple feats b/w nodes
def copy_multiple_u(feat_names):
    return lambda edges: {f_name : edges.src[f_name] for f_name in feat_names}