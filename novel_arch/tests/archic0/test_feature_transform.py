from novel_arch.archic_0.feature_transform import GrambowFeaturizer
from bondnet.utils import seed_torch
import torch

# check that dbond, global features match expected
def test_grambow_featurizer():
    assert False # finish you loser, come back here!

    seed_torch() # same result every time!

    # atom_feat_size = 5
    # atom_count = 10
    # bond_feat_size = 7
    # bond_count = 15
    # global_feat_size = 6
    atom_feat = torch.tensor([(1, 2, 5), (-1, 6, -4)])
    bond_feat = torch.randn([bond_count, bond_feat_size])
    global_feat = torch.randn([1, global_feat_size])
    feats = {
        'atom': atom_feat,
        'bond': bond_feat,
        'global': global_feat,
    }
    featurizer = GrambowFeaturizer(atom_feat_size, bond_feat_size)
    
