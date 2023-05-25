### calculations for d_bond, global node updates
import dgl
from torch.nn import functional as F
import torch
from bondnet.utils import seed_torch

from novel_arch.dmpnn_like.directed_conv import DMPNNPropag

def test_dmpnnpropag_features_no_norm():
    # seed_torch() # same result every time!

    # graph is that atoms are omitted, bonds are now directionally
    # attached to other bonds as per d-mpnn requirements
    # and all connected to global node
    # 0 1 2 d_bonds will point away from center
    # 3 4 5 point to center, n-3 will be original atoms pointing from, or reversed of previous three d_bond
    # (ie all bonds pointing in)
    db2db = ([3,3, 4,4, 5,5], [1,2, 0,2, 0,1])
    db2g = ([0,1,2,3,4,5], [0,0,0,0,0,0])
    g = dgl.heterograph({
        ('d_bond', 'db2db', 'd_bond') : db2db,
        ## one of the following may be unneeded
        # if we prevent propagation one way or another
        ('d_bond', 'db2g', 'global') : db2g,
        ('global', 'g2db', 'd_bond') : tuple(reversed(db2g)),
        # self loop global
        # ('global', 'g2g', 'global') : ([0], [0])
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
    # glob_feat = torch.randn([1, feat_size])
    feats = {
        'd_bond': db_feat,
        # 'global': glob_feat,
    }
    initial_feats = {
        'd_bond': torch.randn([6, feat_size]),
    }

    out_size = feat_size
    gated_conv = DMPNNPropag(feat_size, out_size, residual=False, batch_norm=False)

    # def att_mech(db, pred, preds):
    #     att_sum = torch.sum(torch.stack(tuple(map(lambda p: F.sigmoid(gated_conv.db_att_indiv(db_feat[p])), preds))), dim=0)
    #     denom = att_sum + 1e-6
    #     return F.sigmoid(gated_conv.db_att_indiv(db_feat[pred])) / denom
    def aggreg_db_neighbors(db): # aggregation function for above type of graph
        preds = g.predecessors(db, 'db2db')
        if len(preds) == 0:
            return torch.zeros(3)
        # return sum(map(lambda pred: torch.mul(att_mech(db, pred, preds), gated_conv.db_aggreg_indiv(db_feat[pred])), preds))
        return sum(map(lambda pred: db_feat[pred], preds))
        # return torch.sum(torch.stack(tuple(map(lambda pred: gated_conv.db_aggreg_indiv(db_feat[pred]), preds))), dim=0) # NOTE: change this back!

    db_expected = F.relu(
        gated_conv.message_prop(
            torch.stack(tuple(map(lambda db: 
                aggreg_db_neighbors(db)
            , range(g.num_nodes('d_bond')))))
        ) + initial_feats['d_bond'] ) #+ db_feat
    # glob_expected = F.relu(
    #     torch.mean(
    #         torch.stack(tuple(map(lambda db: gated_conv.H_glob_db(db_expected[db]), range(g.num_nodes('d_bond')))))
    #         , dim=0) 
    # + gated_conv.I_glob_glob(glob_feat)) + glob_feat
    EXPECTED = {
        'd_bond': db_expected,
        # 'global': glob_expected,
    }

    actual = gated_conv(g, feats, initial_feats)

    assert torch.all(torch.isclose(actual['d_bond'], EXPECTED['d_bond'])).item()
    # assert torch.all(torch.isclose(actual['global'], EXPECTED['global'])).item()

# ensure a lone global in a graph undergoes expected transformation
# which is relevant to single atom molecule case, where dmpnn graph will have one global
def DONT_test_directed_conv_lone_global():
    g = dgl.heterograph({
        # self loop global
        ('global', 'g2g', 'global') : ([0], [0]) # just to create global node
    })
    dgl.remove_edges(g, [0], etype='g2g') # make it represent actual case

    feat_size = 3 # unified for global and d_bond once graph is sent into model, so conv op sees unified feature size
    # db_feat = torch.randn([6, feat_size])
    glob_feat = torch.randn([1, feat_size])
    feats = {
        # 'd_bond': db_feat,
        'global': glob_feat,
    }

    out_size = feat_size
    gated_conv = GatedGCNConvDMPNN(feat_size, out_size, residual=True, batch_norm=False)

    glob_expected = F.relu(gated_conv.I_glob_glob(glob_feat)) + glob_feat
    EXPECTED = {
        'global': glob_expected,
    }

    actual = gated_conv(g, feats)

    assert torch.all(torch.isclose(actual['global'], EXPECTED['global'])).item()