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
        atom_2_db = dbond_to_atom_map(dmpnn_g)
        atom_precurs = gen_final_atom_precurs(feats, dmpnn_g, original_feats, atom_2_db)
        # atom_precurs = torch.randn([4, 14])
        atom_feats = self.relu(self.map(atom_precurs))

        feats = {
            'global': feats['global'],
            'atom': atom_feats,
        }
        return feats

# aggregate features of bonds going into atom
def dbond_aggregate(atom, feats, atom_2_db):
    incoming_db_feat = map(lambda db: feats['d_bond'][db], atom_2_db[atom])
    return sum(incoming_db_feat)

# return dict that has index as atom, entry as all dbond pointing to that atom
def dbond_to_atom_map(dmpnn_g):
    atom_2_db = {}
    for db in range(dmpnn_g.num_nodes('d_bond')):
        dest_atom = dmpnn_g.nodes['d_bond'].data['dest_atom'][db].item()
        if dest_atom not in atom_2_db:
            atom_2_db[dest_atom] = []
        atom_2_db[dest_atom].append(db)
    return atom_2_db

# construct final atom precursor matrix - each row is for each atom associated dbond features and initial atom features
def gen_final_atom_precurs(feats, dmpnn_g, original_feats, atom_2_db):
    atom_initial_size = len(original_feats['atom'][0])
    dbond_size = len(feats['d_bond'][0])

    total_atoms = len(original_feats['atom'])
    atom_precurs = torch.empty([total_atoms, atom_initial_size + dbond_size])
    for a in range(total_atoms):
        atom_precurs[a] = torch.cat([original_feats['atom'][a], dbond_aggregate(a, feats, atom_2_db)])

    return atom_precurs