from novel_arch.deep_attn.feat_type_updaters import EdgeNeighborUpdate
import dgl
import torch

g = dgl.heterograph({
    ('atom', 'a2b', 'bond') : ([0, 1], [0, 0]),
    ('global', 'g2b', 'bond') : ([0], [0]),
})

feats = {
    'atom' : torch.tensor([[1,2],[5,3]], dtype=torch.float),
    'bond' : torch.tensor([[-1,3]], dtype=torch.float),
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

thing = EdgeNeighborUpdate({'bond': 2, 'atom': 2, 'global':2}, 32)
print(thing(feats, g))