from torch import nn
import torch
from dgl import function as fn
import dgl
from itertools import pairwise, chain

class EdgeNeighborUpdate(nn.Module):
    ''' 
        MLP update of bond, atom, global features
    '''
    def __init__(self, in_mlp_size, out_mlp_size, inner_layer_sizes=[], residual=False, bias=False):
        super().__init__()

        self.residual = residual
        layer_sizes = [in_mlp_size] + inner_layer_sizes + [out_mlp_size]
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
    def __init__(self):
        super().__init__()

    def forward(self, feats, graph):
        g = graph.local_var()

        # g.nodes['atom'].update({'idx': torch.})

        # store atom indices
        g.nodes['atom'].data.update({'i' : torch.arange(g.num_nodes('atom')).float()})
        # atom features to bond transfer for edge + atom agggregation
        g.update_all(fn.copy_u("ft", "ft_a"), copy_mailbox_feat('ft_a'), etype="a2b")
        g.update_all(fn.copy_u("i", "i_a"), copy_mailbox_feat('i_a'), etype="a2b")

        g.multi_update_all(
            {
                'b2a' : (copy_multiple_u(['ft_a', 'i_a', 'ft']), sum_atom_edge_feat), # this is where we'd be changing the aggregation for atoms
                'g2a' : (fn.copy_u('ft', 'm'), fn.sum('m', 'g')),
            },
            'sum'
        )

        # finish graph processing via concatenation and mlp

        return g

def sum_atom_edge_feat(nodes):
    return aggreg_atom_edge_feat(nodes, lambda t: torch.sum(t, dim=1))

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

    concat_a_b = torch.cat([non_self_ft_a, bond_ft], dim=-1)
    aggregated = aggreg(concat_a_b)

    # print('nodes', nodes.nodes())
    # print('idx of atoms', i_a.shape)
    # print('non self ft a', non_self_ft_a.shape)
    # print('bond feat', bond_ft.shape)
    # print('combined feat', concat_a_b.shape)
    # print('aggerg', aggregated.shape)
    # print('feats of atoms', ft_a.shape)
    # print('omitted attempt i of atom', torch.tensor(omitted_self_i_a).shape)

    return {'a_b_aggreg' : aggregated}

# return reducer fn to just plant full copy of mailbox as feat
def copy_mailbox_feat(feat_name):
    return lambda nodes: {feat_name : nodes.mailbox[feat_name]}

# propagate multiple feats b/w nodes
def copy_multiple_u(feat_names):
    return lambda edges: {f_name : edges.src[f_name] for f_name in feat_names}