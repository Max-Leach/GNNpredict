import csv

from novel_arch.deep_attn.data.dataset import BDEDataset
# from novel_arch.deep_attn.data.dataloader import DataLoader
from novel_arch.deep_attn.data.initial_containers import DirectSmilesRepo, DGLwBDEMappings
from novel_arch.deep_attn.data.featurizers import AtomFeaturize, BondFeaturize, GlobalFeaturize

def bdedataset_from_csv(path, max_lines=None, entry_name_to_col={'reacs': [1], 'prods': [3,4], 'broken_idx': 2, 'bde': 5}, **kwargs):
    with open(path) as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        line_count = 0
        dsr = DirectSmilesRepo()
        for rxn_line in csv_read:
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
        dset = BDEDataset(dsr, bdemap, featurizers={'atom' : AtomFeaturize(aprop, [1, 6, 7, 8]), 'bond' : BondFeaturize(bprop), 'global' : GlobalFeaturize(gprop),}, **kwargs)

        return dset