import torch

class AtomFeaturize:
    def __init__(self, properties: set, avail_atomic_nums):
        avail_feature_gens = {
            'atomic_num': lambda a, m: one_hot_of_set(a.GetAtomicNum(), avail_atomic_nums),
            'total_degree': lambda a, m: [a.GetTotalDegree()],
            'is_in_ring': lambda a, m: [a.IsInRing()],
            'total_num_hs': lambda a, m: [a.GetTotalNumHs(includeNeighbors=True)],
            'ring_of_size': lambda a, m: [m.GetRingInfo().IsAtomInRingOfSize(a.GetIdx(), size) for size in [3, 4, 5, 6, 7]],
        } # populated with all options
        self.feature_gens = {fn : avail_feature_gens[fn] for fn in properties}
    
    def __call__(self, mol):
        feat_compounded = [[self.feature_gens[fn](a, mol) for fn in self.feature_gens.keys()] for a in mol.GetAtoms()]
        return torch.tensor([[ft for sub in atom_ft for ft in sub] for atom_ft in feat_compounded])

# given a list of options, encode one hot of entry
def one_hot_of_set(entry, avail):
    return [1 if entry == e else 0 for e in avail]