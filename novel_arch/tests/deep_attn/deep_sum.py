from novel_arch.deep_attn.model import DeepAtomSum
import dgl
from dgl import function as fn
import torch

a2b = ([0,0,0, 1,2,3], [0,1,2, 0,1,2])
g2b = ([0,0,0], [0,1,2])
g2a = ([0,0,0,0], [0,1,2,3])
g = dgl.heterograph({
    # these three, if they exist, are just self-loops
    ('bond', 'b2b', 'bond') : (torch.arange(3), torch.arange(3)),
    ('atom', 'a2a', 'atom') : (torch.arange(4), torch.arange(4)),
    ('global', 'g2g', 'global') : ([0],[0]),
    
    ('bond', 'b2a', 'atom') : tuple(reversed(a2b)),
    ('atom', 'a2b', 'bond') : a2b, 
    ('global', 'g2b', 'bond') : g2b,
    ('bond', 'b2g', 'global') : tuple(reversed(g2b)),
    ('global', 'g2a', 'atom') : g2a,
    ('atom', 'a2g', 'global') : tuple(reversed(g2a)),
})
feats = {
    'bond': torch.tensor([[1, 2, 3], [3,4,1], [43.4, 3.2,90]], dtype=torch.float),
    'atom': torch.tensor([[43.4, 3.2], [.4, 3.], [44, 3], [33.4, -1.9]], dtype=torch.float),
    'global': torch.tensor([3,23.42], dtype=torch.float).view(1, -1),
}

thing = DeepAtomSum({'bond': 3, 'atom': 2, 'global': 2})
print(thing(feats, g))