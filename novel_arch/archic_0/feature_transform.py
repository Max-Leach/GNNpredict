# feature transforms for bondnet graph to archic 0 type input
from torch import nn
import torch

# exactly the same as in Grambow paper: bond, atom -> directed bond feature
# but now global features are simply maintained
# -for d_bond feat row, does relu(W*(hv||evw))
class GrambowFeaturizer(nn.Module):
    def __init__(self, atom_feat_size, bond_feat_size, out_feat_size):
        super().__init__()

        self.relu = nn.ReLU()
        combined_size = atom_feat_size + bond_feat_size
        self.map = nn.Linear(combined_size, out_feat_size)

    def forward(self, feats, dmpnn_g):
        dbond_precurs = gen_directed_bond_precurs(feats, dmpnn_g)
        d_bond_feats = self.relu(self.map(dbond_precurs))

        feats = {
            'global': feats['global'],
            'd_bond': d_bond_feats,
        }
        return feats

# construct directed bond precursor matrix - each row is for each d bond associated atom and old bond features from bondnet style graph
def gen_directed_bond_precurs(feats, dmpnn_g):
    assert len(feats['atom']) > 0 and len(feats['bond']) > 0
    atom_feat_size = len(feats['atom'][0])
    bond_feat_size = len(feats['bond'][0])
    # in place insertion
    dbond_precurs = torch.empty([dmpnn_g.num_nodes('d_bond'), atom_feat_size + bond_feat_size])

    for db in range(len(dbond_precurs)):
        # print("src atom", dmpnn_g.ndata['src_atom'])
        src_atom = dmpnn_g.nodes['d_bond'].data['src_atom'][db]
        old_bond = dmpnn_g.nodes['d_bond'].data['old_bond'][db]
        dbond_precurs[db] = torch.cat([feats['atom'][src_atom], feats['bond'][old_bond]])

    return dbond_precurs