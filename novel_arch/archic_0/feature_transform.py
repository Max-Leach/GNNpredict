# feature transforms for bondnet graph to archic 0 type input
from torch import nn
import torch

# exactly the same as in Grambow paper: bond, atom -> directed bond feature
# but now global features are simply maintained
# -for d_bond feat row, does relu(W*(hv||evw))
class GrambowFeaturizer(nn.Module):
    def __init__(self, atom_feat_size, bond_feat_size):
        super().__init__()

        self.relu = nn.ReLU()
        combined_size = atom_feat_size + bond_feat_size
        self.map = nn.Linear(combined_size, combined_size)

    def forward(self, feats):
        dbond_precurs = gen_directed_bond_precurs(feats)
        d_bond_feats = self.relu(self.map(dbond_precurs))

        # atom_bond = torch.cat([feats['atom'], feats['bond']], dim=-1)
        # dbond = self.relu(self.map(atom_bond))

        feats = {
            'global': feats['global'],
            'd_bond': d_bond_feats,
        }
        return feats

# construct directed bond precursor matrix - each row is for each d bond associated atom and old bond features from bondnet style graph
def gen_directed_bond_mat(feats):
    # we not only need the graph to do this, we need to know which
    # atom and bond is associated to each directed edge!

    assert False # finish this loser