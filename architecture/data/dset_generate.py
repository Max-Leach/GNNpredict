import csv

from architecture.data.dataset import BDEDataset
from architecture.data.initial_containers import DirectSmilesRepo, DGLwBDEMappings
from architecture.data.featurizers import AtomFeaturize, BondFeaturize, GlobalFeaturize

def from_csv(path, max_lines=None, start_line=None, entry_name_to_col={'reacs': [1], 'prods': [3,4], 'broken_idx': 2, 'bde': 5, 'bondtype': 6}, **kwargs):
    if start_line == None:
        start_line = 0
    # atomic_num_list = get_atomic_num_list(path, max_lines, start_line, entry_name_to_col['bondtype'])
    ## do below so results are more repeatable for users - above version will vary atomic features based on whats available in dataset and if a user only does a subset, the atomic feature size will be different than what deepbde expects
    atomic_num_list = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17]

    with open(path) as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        line_count = 0
        dsr = DirectSmilesRepo()
        for rxn_line in tuple(csv_read)[start_line:]:
            rsmiles = [rxn_line[i] for i in entry_name_to_col['reacs']]
            psmiles = [rxn_line[i] for i in entry_name_to_col['prods']]
            broken_bond_idxs = [int(rxn_line[entry_name_to_col['broken_idx']])]
            bde = float(rxn_line[entry_name_to_col['bde']])
            dsr.append_reaction(rsmiles, psmiles, broken_bond_idxs, bde)
            line_count += 1
            if max_lines != None and line_count >= max_lines:
                break
        bdemap = DGLwBDEMappings(dsr)

        aprop = ['atomic_num', 'total_degree', 'total_num_hs', 'ring_of_size', 'is_in_ring']
        bprop = ['is_in_ring', 'ring_of_size', 'dative']
        gprop = ['num_atoms', 'num_bonds', 'total_weight']
        dset = BDEDataset.from_initials(dsr, bdemap, featurizers={'atom' : AtomFeaturize(aprop, atomic_num_list), 'bond' : BondFeaturize(bprop), 'global' : GlobalFeaturize(gprop),}, **kwargs)

        return dset

from rdkit.Chem import rdchem
def get_atomic_num_list(path, max_lines, start_line, bond_type_col):
    atom_types = set()
    with open(path) as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for rxn_line in tuple(csv_read)[start_line:]:
            bt = rxn_line[bond_type_col]
            bt = bt.split('-')
            for a in bt:
                atom_types.add(rdchem.Atom(a).GetAtomicNum())
            line_count += 1
            if max_lines != None and line_count >= max_lines:
                break
    return sorted(atom_types)