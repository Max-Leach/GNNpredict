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
        layer_sizes = [in_mlp_size] + inner_layer_sizes + [out_size]
        fc_nested = [(nn.Linear(fc_lens[0], fc_lens[1], bias=bias), nn.ReLU()) for fc_lens in pairwise(layer_sizes)]
        self.fc = nn.Sequential(*tuple(chain(*fc_nested)))

    def forward(self, feats, graph): # features are assumed to be loaded in before this fn
        g = graph.local_var()

        # g.nodes['bond'].update({'b': feats['bond']})
        # g.nodes['atom'].update({'a': feats['atom']})
        # g.nodes['global'].update({'g': feats['global']})

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
            b += feats['bond']
        feats['bond'] = b
        return feats

class AtomAggregUpdate(nn.Module):
    '''
        custom aggregation of node features (especially bonds)
    '''
    def __init__(self, atom_edge_feat_aggreg, in_feat_sizes, out_size, inner_layer_sizes=[], residual=False, bias=False):
        super().__init__()

        self.atom_edge_feat_aggreg = atom_edge_feat_aggreg
        self.residual = residual
        in_mlp_size = 2*in_feat_sizes['atom']+in_feat_sizes['global']+in_feat_sizes['bond']
        self.fc = mlp_from_sizes(in_mlp_size, out_size, inner_layer_sizes, bias=bias)

    def forward(self, feats, graph):
        g = graph.local_var()

        # store atom indices
        g.nodes['atom'].data.update({'i' : torch.arange(g.num_nodes('atom')).float()})
        # atom features to bond transfer for edge + atom agggregation
        g.update_all(fn.copy_u("ft", "ft_a"), copy_mailbox_feat_repeat_if_single('ft_a'), etype="a2b")
        g.update_all(fn.copy_u("i", "i_a"), copy_mailbox_feat_repeat_if_single('i_a'), etype="a2b")

        # g.multi_update_all(
        #     {
        #         'b2a' : (copy_multiple_u(['ft_a', 'i_a', 'ft']), concat_sum_atom_edge_feat), # this is where we'd be changing the aggregation for atoms
        #         'g2a' : (fn.copy_u('ft', 'm'), fn.sum('m', 'g')),
        #     },
        #     'sum'
        # )

        g.update_all(copy_multiple_u(['ft_a', 'i_a', 'ft']), self.atom_edge_feat_aggreg, etype='b2a') # this is where we'd be changing the aggregation for atoms
        g.update_all(fn.copy_u('ft', 'm'), fn.sum('m', 'g'), etype='g2a')

        # finish graph processing via concatenation and mlp
        a_b_aggreg = g.nodes['atom'].data['a_b_aggreg']
        fc_in = torch.cat([feats['atom'], a_b_aggreg, g.nodes['atom'].data['g']], dim=-1)
        a = self.fc(fc_in)
        if self.residual:
            a += feats['atom']
        feats['atom'] = a

        return feats

def concat_sum_atom_edge_feat(nodes):
    return aggreg_atom_edge_feat(nodes, lambda a,b: torch.sum(torch.cat([a, b], dim=-1), dim=1))

# aggregate incoming atom feat concat with bond feat that connects them
def aggreg_atom_edge_feat(nodes, aggreg):
    ft_a = nodes.mailbox['ft_a']
    i_a = nodes.mailbox['i_a'] # order of idx of atoms should correspond with atom feats of ft_a
    bond_ft = nodes.mailbox['ft']

    # these operations below are done manually because i couldn't seem to find any torch methods that
    # did quite what I needed wihout some weird tensor manipulations

    # indices into feat that aren't eh self of atom
    omitted_self_i_a = []
    for self_i, bonds_idxs in zip(nodes.nodes(), i_a):
        omitted_self_i_a.append([])
        for bond_idxs in bonds_idxs:
            if bond_idxs[0].item() != self_i:
                # omitted_self_i_a[-1].append(bond_idxs[0].item())
                omitted_self_i_a[-1].append(0)
            else:
                # omitted_self_i_a[-1].append(bond_idxs[1].item())
                omitted_self_i_a[-1].append(1)


    # non_self_ft_a = ft_a[torch.tensor(omitted_self_i_a)]
    # select features
    non_self_ft_a = []
    for at_i, ft_idxs in enumerate(omitted_self_i_a):
        at_fts = []
        for incoming_idx, ft_idx in enumerate(ft_idxs):
            at_fts.append(ft_a[at_i][incoming_idx][ft_idx])
        non_self_ft_a.append(torch.stack(at_fts))
    non_self_ft_a = torch.stack(non_self_ft_a)

    # concat_a_b = torch.cat([non_self_ft_a, bond_ft], dim=-1)
    aggregated = aggreg(non_self_ft_a, bond_ft)

    # print('nodes', nodes.nodes())
    # print('idx of atoms', i_a.shape)
    # print('non self ft a', non_self_ft_a.shape)
    # print('bond feat', bond_ft.shape)
    # print('combined feat', concat_a_b.shape)
    # print('aggerg', aggregated.shape)
    # print('feats of atoms', ft_a.shape)
    # print('omitted attempt i of atom', torch.tensor(omitted_self_i_a).shape)

    return {'a_b_aggreg' : aggregated}

class GlobalAggregUpdate(nn.Module):
    def __init__(self, edge_aggreg, atom_aggreg, in_feat_sizes, out_size, inner_layer_sizes=[], residual=False, bias=False):
        super().__init__()

        self.atom_aggreg = atom_aggreg
        self.edge_aggreg = edge_aggreg
        self.residual = residual
        in_mlp_size = sum(in_feat_sizes.values())
        self.fc = mlp_from_sizes(in_mlp_size, out_size, inner_layer_sizes, bias=bias)

    def forward(self, feats, graph):
        g = graph.local_var()

        # fn.mean('m', 'a')
        # fn.mean('m', 'b')
        g.update_all(fn.copy_u('ft', 'm'), self.atom_aggreg, etype='a2g')
        g.update_all(fn.copy_u('ft', 'm'), self.edge_aggreg, etype='b2g')

        a = g.nodes['global'].data['a']
        b = g.nodes['global'].data['b']
        fc_in = torch.cat([feats['global'], b, a], dim=-1)
        glob = self.fc(fc_in)
        if self.residual:
            glob += feats['global']
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