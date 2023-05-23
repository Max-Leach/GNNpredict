import dgl
from torch.nn import functional as F
import torch

from novel_arch.archic_0.readout_feature_transform import DBondtoAtomFeaturize

def test_dbond_to_atom_featurize():
    # seed_torch() # same result every time!

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

    # load dest atom feature in dmpnn graph
    dest_atom_feat = torch.zeros(6, dtype=torch.int)
    for db in range(len(dest_atom_feat)):
        old_bond = dmpnn_g.nodes['d_bond'].data['old_bond'][db]
        src_atom = dmpnn_g.nodes['d_bond'].data['src_atom'][db]
        dest_a = tuple(filter(lambda a: a != src_atom.item(), orig_g.predecessors(old_bond.item(), etype='a2b').tolist()))[0]
        dest_atom_feat[db] = dest_a
    dmpnn_g.nodes['d_bond'].data['dest_atom'] = dest_atom_feat

    # dbond features
    dbond_feat_size = 11
    dbond_feat = torch.rand([6, dbond_feat_size]) + 1 # prevent 0 case
    feats = {
        'global': torch.randn([3, 3]),
        'd_bond': dbond_feat,
    }
    
    # precursor to get expected atom features
    atom_precurs = torch.stack(tuple(
        map(
            lambda a: torch.cat([ORIG_FEATS['atom'][a], dbond_feat[dmpnn_g.nodes['d_bond'].data['dest_atom'][a]]], dim=0)
        , range(orig_g.num_nodes('atom')))
    ))
    
    out_atom_feat_size = 7
    featurizer = DBondtoAtomFeaturize(atom_feat_size=3, bond_feat_size=dbond_feat_size, out_atom_feat_size=out_atom_feat_size)
    
    EXPECTED = {
        'atom': F.relu(featurizer.map(atom_precurs)),
        'global': feats['global']
    }

    actual = featurizer(feats, dmpnn_g, ORIG_FEATS)

    assert len(EXPECTED.keys()) > 0
    for ntype in EXPECTED:
        assert torch.all(torch.isclose(actual[ntype], EXPECTED[ntype])).item() 