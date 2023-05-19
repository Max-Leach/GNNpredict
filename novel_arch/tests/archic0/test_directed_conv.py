### calculations for d_bond, global node updates
import dgl
from torch.nn import functional as F
import torch
from bondnet.utils import seed_torch

from novel_arch.archic_0.directed_conv import GatedGCNConvDMPNN

def test_directed_conv_features_residue_no_norm():
    seed_torch() # same result every time!

    # graph is that atoms are omitted, bonds are now directionally
    # attached to other bonds as per d-mpnn requirements
    # and all connected to global node
    # 0 1 2 d_bonds will point away from center
    # 3 4 5 point to center, n-3 will be original atoms pointing from, or reversed of previous three d_bond
    # (ie all bonds pointing in)
    db2db = ([3,3, 4,4, 5,5,  0,1,2,3,4,5], [1,2, 0,2, 0,1,  0,1,2,3,4,5])
    db2g = ([0,1,2,3,4,5], [0,0,0,0,0,0])
    g = dgl.heterograph({
        ('d_bond', 'db2db', 'd_bond') : db2db,
        ## one of the following may be unneeded
        # if we prevent propagation one way or another
        ('d_bond', 'db2g', 'global') : db2g,
        ('global', 'g2db', 'd_bond') : tuple(reversed(db2g)),
        # self loop global
        ('global', 'g2g', 'global') : ([0], [0])
    })
    ## previous graph referencing
    # g.nodes['d_bond'].data['src_atom'] = torch.tensor(
    #         [0,0,0, # first three d_bonds point away from center
    #         1,2,3] # last three point into center, away from outsiders
    #     , dtype=torch.int)
    # g.nodes['d_bond'].data['old_bond'] = torch.tensor(
    #         [0,1,2, 0,1,2] # in and out triplets correspond to old bonds in same order
    #     , dtype=torch.int)

    # features for graph
    feat_size = 3 # unified for global and d_bond once graph is sent into model, so conv op sees unified feature size
    db_feat = torch.randn([6, feat_size])
    glob_feat = torch.randn([1, feat_size])
    feats = {
        'd_bond': db_feat,
        'global': glob_feat,
    }

    out_size = 3
    gated_conv = GatedGCNConvDMPNN(feat_size, out_size, residual=True)

    def att_mech(db, pred, preds):
        att_sum = sum(map(lambda p: F.sigmoid(gated_conv.db_att_indiv(db_feat[p])), preds))
        denom = att_sum + gated_conv.tol
        return F.sigmoid(gated_conv.db_att_indiv(db_feat[pred])) / denom
    def aggreg_db_neighbors(db): # aggregation function for above type of graph
        preds = g.predecessors(db, 'db2db')
        return sum(map(lambda pred: torch.mul(att_mech(db, pred, preds), db_feat[pred]), preds))

    db_expected = torch.cat(tuple(map(lambda db: 
        gated_conv.B_db_db(db_feat[db]) + 
        gated_conv.db_aggreg(aggreg_db_neighbors(db)) + 
        gated_conv.C_db_glob(glob_feat), 
        range(g.num_nodes('d_bond'))))) + db_feat
    glob_expected = F.relu(
        sum(map(lambda db: gated_conv.H_glob_db(db_feat[db]), range(g.num_nodes('d_bond')))) 
    + gated_conv.I_glob_glob(glob_feat)) + glob_feat
    EXPECTED = {
        'd_bond': db_expected,
        'global': glob_expected,
    }

    actual = gated_conv(g, feats)

    for k in EXPECTED:
        assert torch.all(torch.isclose(actual[k], EXPECTED[k])).item()