from novel_arch.deep_attn.feat_type_updaters import AtomAggregUpdate, concat_sum_atom_edge_feat
import dgl
import torch

a2b = ([2,0,1,1, 3,1], [1,0,0,1, 2,2])
g2a = ([0,0,0,0], [0,1,2,3])
g = dgl.heterograph({
    ('atom', 'a2b', 'bond') : a2b,
    ('bond', 'b2a', 'atom') : tuple(reversed(a2b)),
    ('global', 'g2a', 'atom') : g2a,
})

feats = {
    'atom' : torch.tensor([[1,2],[5,3],[6,9],[-2,4]], dtype=torch.float),
    'bond' : torch.tensor([[-1,3,2], [-9,2,7.3], [8,3,-.34]], dtype=torch.float),
    'global' : torch.tensor([[3,4]], dtype=torch.float),
}
g.nodes['atom'].data.update({
    'ft' : feats['atom']
})
g.nodes['bond'].data.update({
    'ft' : feats['bond']
})
g.nodes['global'].data.update({
    'ft' : feats['global']
})

thing = AtomAggregUpdate(concat_sum_atom_edge_feat, {'bond': 3, 'atom': 2, 'global':2}, 69, inner_layer_sizes=[4,3], bias=False)
print(thing(feats, g))