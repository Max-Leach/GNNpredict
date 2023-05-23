from torch import nn
import torch

# convert form directed bond features and initial atom features
# to a new atom feature for readout phase
class DBondtoAtomFeaturize(nn.Module):
    def __init__(self, atom_feat_size, bond_feat_size, out_atom_feat_size):
        super().__init__()

        self.relu = nn.ReLU()
        combined_size = atom_feat_size + bond_feat_size
        self.map = nn.Linear(combined_size, out_atom_feat_size)

    def forward(self, feats, dmpnn_g, original_feats):
        # atom_precurs = gen_final_atom_precurs(feats, dmpnn_g, original_feats)
        atom_precurs = torch.randn([4, 14])

        atom_feats = self.relu(self.map(atom_precurs))

        feats = {
            'global': feats['global'],
            'atom': atom_feats,
        }
        return feats

# construct final atom precursor matrix - each row is for each atom associated dbond features and initial atom features
def gen_final_atom_precurs(feats, dmpnn_g, original_feats):
    # NOTE: DO YOUR WORK IN THIS FN, INITIAL ATOM + DBOND FEATURES AGGREGATED!

    assert len(feats['atom']) > 0 and len(feats['bond']) > 0
    atom_feat_size = len(feats['atom'][0])
    bond_feat_size = len(feats['bond'][0])
    # in place insertion
    dbond_precurs = torch.empty([dmpnn_g.num_nodes('d_bond'), atom_feat_size + bond_feat_size])

    for db in range(len(dbond_precurs)):
        src_atom = dmpnn_g.nodes['d_bond'].data['src_atom'][db]
        old_bond = dmpnn_g.nodes['d_bond'].data['old_bond'][db]
        dbond_precurs[db] = torch.cat([feats['atom'][src_atom], feats['bond'][old_bond]])

    return dbond_precurs