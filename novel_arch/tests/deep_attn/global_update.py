from novel_arch.deep_attn.feat_type_updaters import GlobalAggregUpdate
import dgl
from dgl import function as fn
import torch

a2g = ([0, 1, 2, 3], [0, 0, 0, 0])
b2g = ([0, 1, 2], [0, 0, 0])
g = dgl.heterograph({
    ('atom', 'a2g', 'global') : a2g,
    ('bond', 'b2g', 'global') : b2g,
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

thing = GlobalAggregUpdate(fn.mean('m', 'b'), fn.mean('m', 'a'), {'bond': 3, 'atom': 2, 'global':2})
print(thing(feats, g))