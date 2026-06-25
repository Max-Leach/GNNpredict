import torch
from rdkit.Chem import rdchem
from rdkit import Chem
import networkx as nx
import numpy as np
from architecture.data.nsppk import NodeNSPPK
from scipy.sparse import vstack, csr_matrix


class AtomFeaturize:
    ## properties - list of properties from avail generators in constructor
    # avail_atom_nums - list of atomic numbers, depends on dataset
    def __init__(self, properties, avail_atomic_nums):
        avail_feature_gens = {
            'atomic_num': lambda a, m: one_hot_of_set(a.GetAtomicNum(), avail_atomic_nums),
            'total_degree': lambda a, m: [a.GetTotalDegree()],
            'is_in_ring': lambda a, m: [int(a.IsInRing())],
            'total_num_hs': lambda a, m: [a.GetTotalNumHs(includeNeighbors=True)],
            'ring_of_size': lambda a, m: [int(m.GetRingInfo().IsAtomInRingOfSize(a.GetIdx(), size)) for size in
                                          [3, 4, 5, 6, 7]],
        }
        self.feature_gens = {fn: avail_feature_gens[fn] for fn in properties}

    def __call__(self, mol):
        feat_compounded = [[self.feature_gens[fn](a, mol) for fn in self.feature_gens.keys()] for a in mol.GetAtoms()]
        return torch.tensor([[ft for sub in atom_ft for ft in sub] for atom_ft in feat_compounded], dtype=torch.float)


class BondFeaturize:
    ## properties - list of properties from avail generators in constructor
    def __init__(self, properties):
        avail_feature_gens = {
            'is_in_ring': lambda b, m: [int(b.IsInRing())],
            'ring_of_size': lambda b, m: [int(m.GetRingInfo().IsBondInRingOfSize(b.GetIdx(), size)) for size in
                                          [3, 4, 5, 6, 7]],
            'dative': lambda b, m: [int(b.GetBondType() == rdchem.BondType.DATIVE)],
        }
        avail_feature_lens = {  # ensure these values track the feature generators above
            'is_in_ring': 1,
            'ring_of_size': 5,
            'dative': 1,
        }
        self.feature_gens = {fn: avail_feature_gens[fn] for fn in properties}
        self.b_len = sum([avail_feature_lens[prop] for prop in
                          properties])  # for no bond case - add phantom bond with this shape that should match above

    def __call__(self, mol):
        if mol.GetNumBonds() == 0:
            return torch.zeros(1, self.b_len, dtype=torch.float)
        feat_compounded = [[self.feature_gens[fn](b, mol) for fn in self.feature_gens.keys()] for b in mol.GetBonds()]
        return torch.tensor([[ft for sub in bond_ft for ft in sub] for bond_ft in feat_compounded], dtype=torch.float)


class GlobalFeaturize:
    def __init__(self, properties, nsppk_params=None):
        """
        Initialize GlobalFeaturize with optional NSPPK parameters.

        Args:
            properties: List of global property names to compute
            nsppk_params: Dict with NSPPK parameters {'radius', 'distance', 'connector', 'nbits', 'sigma'}
                         If None, NSPPK features won't be computed
        """
        self.pt = rdchem.GetPeriodicTable()
        avail_feature_gens = {
            'num_atoms': lambda m: [m.GetNumAtoms()],
            'num_bonds': lambda m: [m.GetNumBonds()],
            'total_weight': lambda m: [sum([self.pt.GetAtomicWeight(a.GetAtomicNum()) for a in m.GetAtoms()])],
        }
        self.feature_gens = {fn: avail_feature_gens[fn] for fn in properties}
        self.nsppk_params = nsppk_params

        # Initialize NSPPK vectorizer if parameters are provided
        if nsppk_params is not None:
            self.nsppk_vectorizer = NodeNSPPK(
                radius=nsppk_params.get('radius', 1),
                distance=nsppk_params.get('distance', 1),
                connector=nsppk_params.get('connector', 1),
                nbits=nsppk_params.get('nbits', 5),
                dense=False,
                sigma=nsppk_params.get('sigma', 1)
            )
        else:
            self.nsppk_vectorizer = None

    def __call__(self, mol, canon_smiles=None, broken_bond_idx=None):
        """
        Compute global features, optionally including NSPPK features.

        Args:
            mol: RDKit molecule object
            canon_smiles: Canonical SMILES string (needed for NSPPK)
            broken_bond_idx: Index of broken bond (needed for NSPPK)

        Returns:
            Torch tensor of global features (base + NSPPK if enabled)
        """
        # Compute base global features
        feat_compounded = [[self.feature_gens[fn](mol) for fn in self.feature_gens.keys()]]
        base_feats = torch.tensor(
            [[ft for sub in bond_ft for ft in sub] for bond_ft in feat_compounded],
            dtype=torch.float
        )

        # If NSPPK is enabled, concatenate NSPPK features (zeros if bond info unavailable)
        if self.nsppk_vectorizer is not None:
            nsppk_size = 2 ** self.nsppk_params.get('nbits', 5)
            if canon_smiles is not None and broken_bond_idx is not None:
                nsppk_vec = self._compute_nsppk_features(canon_smiles, broken_bond_idx)
            else:
                nsppk_vec = torch.zeros(1, nsppk_size, dtype=torch.float)
            base_feats = torch.cat([base_feats, nsppk_vec], dim=1)

        return base_feats

    def _compute_nsppk_features(self, smiles, bond_idx):
        """
        Compute NSPPK features for a single molecule.

        Args:
            smiles: SMILES string
            bond_idx: Index of the bond to focus on

        Returns:
            Torch tensor of NSPPK features (1 x feature_dim)
        """
        # Convert SMILES to NetworkX graph
        graph = smiles_to_graph(smiles)

        # Get atom indices for the broken bond
        atom_indices = get_bond_atoms(smiles, bond_idx)

        # Transform graph to get node features
        X_transformed = self.nsppk_vectorizer.fit_transform([graph])

        # Sum features for the two atoms of the broken bond
        product_row_1 = X_transformed[0][atom_indices[0]]
        product_row_2 = X_transformed[0][atom_indices[1]]
        product_sum = product_row_1 + product_row_2

        # Convert to dense tensor
        if hasattr(product_sum, 'toarray'):
            nsppk_array = product_sum.toarray().flatten()
        else:
            nsppk_array = product_sum.flatten()

        return torch.tensor(nsppk_array, dtype=torch.float).unsqueeze(0)


def smiles_to_graph(smiles):
    """Convert SMILES to NetworkX graph for NSPPK processing."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    G = nx.Graph()

    for atom in mol.GetAtoms():
        atom_features = {
            'label': atom.GetSymbol(),
            'vec': np.array([
                float(atom.GetAtomicNum()),
                float(atom.GetFormalCharge()),
                float(atom.GetMass()),
                float(atom.GetHybridization()),
                float(atom.GetNumExplicitHs()),
                float(atom.GetDegree()),
                float(atom.GetIsAromatic()),
                float(atom.IsInRing())
            ])
        }
        G.add_node(atom.GetIdx(), **atom_features)

    for bond in mol.GetBonds():
        bond_features = {'label': str(bond.GetBondType())}
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), **bond_features)

    return G


def get_bond_atoms(smiles, bond_index):
    """Get the atom indices for a given bond."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    bond = mol.GetBondWithIdx(bond_index)
    return bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()


# given a list of options, encode one hot of entry
def one_hot_of_set(entry, avail):
    return [1 if entry == e else 0 for e in avail]