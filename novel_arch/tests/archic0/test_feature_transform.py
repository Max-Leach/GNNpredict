from novel_arch.archic_0.feature_transform import GrambowFeaturizer
from bondnet.utils import seed_torch
import torch
from torch.nn import functional as F
from torch import nn
import dgl

# check that dbond, global features match expected
def test_grambow_featurizer():
    seed_torch() # same result every time!

    # graph is that atoms are omitted, bonds are now directionally
    # attached to other bonds as per d-mpnn requirements
    # and all connected to global node
    # 0 1 2 d_bonds will point away from center
    # 3 4 5 point to center, n-3 will be original atoms pointing from, or reversed of previous three d_bond
    # (ie all bonds pointing in)
    db2db = ([3,3, 4,4, 5,5], [1,2, 0,2, 0,1])
    db2g = ([0,1,2, 3,4,5], [0,2,1, 0,2,1]) # multiple globals
    dmpnn_g = dgl.heterograph({
        ('d_bond', 'db2db', 'd_bond') : db2db,
        ## one of the following may be unneeded
        # if we prevent propagation one way or another
        ('d_bond', 'db2g', 'global') : db2g,
        ('global', 'g2db', 'd_bond') : tuple(reversed(db2g)),
    })
    ## original graph referencing
    dmpnn_g.nodes['d_bond'].data['src_atom'] = torch.tensor(
            [0,0,0, # first three d_bonds point away from center
            1,2,3] # last three point into center, away from outsiders
        , dtype=torch.int)
    dmpnn_g.nodes['d_bond'].data['old_bond'] = torch.tensor(
            [0,1,2, 0,1,2] # in and out triplets correspond to old bonds in same order
        , dtype=torch.int)

    ## corresponds to graph with 3 undirected bonds, 3 atoms, 1 global node
    ## with pairs as so:
    a2b = ([0,0,0, 1,2,3], [0,1,2, 0,1,2])
    g2b = ([0,2,1], [0,1,2]) # multiple globals
    g2a = ([0,0,0,0], [0,1,2,3])
    ORIG_FEATS = {
        'atom': torch.tensor([(0.1, -0.5, 5.0),(5.2, 42.2, -7.2),(42.2, -0.5, -7.2),(1.2, -22.2, -7.2)]),
        'bond': torch.tensor([(5.2, 42.2, -7.2,5.0),(0.1, -0.5, 5.0, -7.2),(8.9, -10.2, 4.0, -2.1)]),
        'global': torch.tensor([(1.0,2.0), (5.2,-9.2), (-8.9, 10.2)]),
    }
    # construct original graph with features to ensure graph data at least (dimensionally) makes sense
    orig_g = dgl.heterograph({
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
    for ntype, feat in ORIG_FEATS.items():
        orig_g.nodes[ntype].data['feat'] = feat

    # concat atom and bond features into matrix for finding expected result
    dbond_concat = torch.stack((
        torch.cat([ORIG_FEATS['atom'][0], ORIG_FEATS['bond'][0]]), # 0 srcd to atom 0, has bond 0
        torch.cat([ORIG_FEATS['atom'][0], ORIG_FEATS['bond'][1]]), # 1 srcd to atom 0, has bond 1
        torch.cat([ORIG_FEATS['atom'][0], ORIG_FEATS['bond'][2]]), # 2 srcd to atom 0, has bond 2
        torch.cat([ORIG_FEATS['atom'][1], ORIG_FEATS['bond'][0]]), # 3 srcd to atom 1, has bond 0
        torch.cat([ORIG_FEATS['atom'][2], ORIG_FEATS['bond'][1]]), # 4 srcd to atom 2, has bond 1
        torch.cat([ORIG_FEATS['atom'][3], ORIG_FEATS['bond'][2]]), # 5 srcd to atom 3, has bond 2
    ))
    assert dbond_concat.shape == (6, len(ORIG_FEATS['atom'][0]) + len(ORIG_FEATS['bond'][0]))
    
    atom_feat_size = len(ORIG_FEATS['atom'][0])
    bond_feat_size = len(ORIG_FEATS['bond'][0])
    out_feat_size = 5
    featurizer = GrambowFeaturizer(atom_feat_size, bond_feat_size, out_feat_size)
    # make linear map in featurizer non-zero, to prevent any 0 case which will make it likelier to pass test erroneously
    featurizer.map.weight = nn.Parameter(torch.rand(featurizer.map.weight.shape) + 1)

    dbond_feats = F.relu(featurizer.map(dbond_concat))
    assert dbond_feats.shape == (6, out_feat_size)
    EXPECTED = {
        'd_bond': dbond_feats,
        'global': ORIG_FEATS['global']
    }

    actual = featurizer(ORIG_FEATS, dmpnn_g)

    # assert torch.all(torch.isclose(actual, EXPECTED))
    assert len(EXPECTED.keys()) > 0
    for ntype in EXPECTED:
        assert torch.all(torch.isclose(actual[ntype], EXPECTED[ntype])).item() 
