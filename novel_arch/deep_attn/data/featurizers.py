import torch
from rdkit.Chem import rdchem

class AtomFeaturize:
    ## properties - list of properties from avail generators in constructor
    # avail_atom_nums - list of atomic numbers, depends on dataset
    def __init__(self, properties, avail_atomic_nums):
        avail_feature_gens = {
            'atomic_num': lambda a, m: one_hot_of_set(a.GetAtomicNum(), avail_atomic_nums),
            'total_degree': lambda a, m: [a.GetTotalDegree()],
            'is_in_ring': lambda a, m: [int(a.IsInRing())],
            'total_num_hs': lambda a, m: [a.GetTotalNumHs(includeNeighbors=True)],
            'ring_of_size': lambda a, m: [int(m.GetRingInfo().IsAtomInRingOfSize(a.GetIdx(), size)) for size in [3, 4, 5, 6, 7]],
        }
        self.feature_gens = {fn : avail_feature_gens[fn] for fn in properties}
    
    def __call__(self, mol):
        feat_compounded = [[self.feature_gens[fn](a, mol) for fn in self.feature_gens.keys()] for a in mol.GetAtoms()]
        return torch.tensor([[ft for sub in atom_ft for ft in sub] for atom_ft in feat_compounded])

# NOTE: haven't seen it work on ringed atoms yet, as current tested dont have rings
class BondFeaturize:
    ## properties - list of properties from avail generators in constructor
    def __init__(self, properties):
        avail_feature_gens = {
            'is_in_ring': lambda b, m: [int(b.IsInRing())],
            'ring_of_size': lambda b, m: [int(m.GetRingInfo().IsBondInRingOfSize(b.GetIdx(), size)) for size in [3, 4, 5, 6, 7]],
            'dative': lambda b, m: [int(b.GetBondType() == rdchem.BondType.DATIVE)],
        }
        avail_feature_lens = { # ensure these values track the feature generators above
            'is_in_ring': 1,
            'ring_of_size': 5, 
            'dative': 1,
        }
        self.feature_gens = {fn : avail_feature_gens[fn] for fn in properties}
        self.b_len = sum([avail_feature_lens[prop] for prop in properties]) # for no bond case - add phantom bond with this shape that should match above

    def __call__(self, mol):
        if mol.GetNumBonds() == 0:
            return torch.zeros(1, self.b_len, dtype=torch.float)
        feat_compounded = [[self.feature_gens[fn](b, mol) for fn in self.feature_gens.keys()] for b in mol.GetBonds()]
        return torch.tensor([[ft for sub in bond_ft for ft in sub] for bond_ft in feat_compounded], dtype=torch.float)

class GlobalFeaturize:
    def __init__(self, properties):
        self.pt = rdchem.GetPeriodicTable()
        avail_feature_gens = {
            'num_atoms': lambda m: [m.GetNumAtoms()],
            'num_bonds': lambda m: [m.GetNumBonds()],
            'total_weight': lambda m: [sum([self.pt.GetAtomicWeight(a.GetAtomicNum()) for a in m.GetAtoms()])],
        }
        self.feature_gens = {fn : avail_feature_gens[fn] for fn in properties}

    def __call__(self, mol):
        feat_compounded = [[self.feature_gens[fn](mol) for fn in self.feature_gens.keys()]]
        return torch.tensor([[ft for sub in bond_ft for ft in sub] for bond_ft in feat_compounded])

# given a list of options, encode one hot of entry
def one_hot_of_set(entry, avail):
    return [1 if entry == e else 0 for e in avail]