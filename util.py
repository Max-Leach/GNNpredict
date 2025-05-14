from architecture.data.rxn_graph import BondDissociate
from architecture.data.initial_containers import DGLwBDEMappings
from architecture.data.single_run_tools import prod_to_reac_atom_map, prod_to_reac_bond_map
from architecture.data.featurizers import AtomFeaturize, BondFeaturize, GlobalFeaturize
from rdkit import Chem
import torch
import dgl
import pickle
import time

with open('../transforms', 'rb') as f:
    transforms = pickle.load(f)

out_mean, out_stdev = transforms['val_mean'], transforms['val_stdev']
stders = transforms['transform']

model_path = '../model'
model = torch.load(model_path, map_location='cpu')
model.eval()

# atomic_num_list = [1, 6, 7, 8]
atomic_num_list = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17]
aprop = ['atomic_num', 'total_degree', 'total_num_hs', 'ring_of_size', 'is_in_ring']
bprop = ['is_in_ring', 'ring_of_size', 'dative']
gprop = ['num_atoms', 'num_bonds', 'total_weight']
featurizers = {'atom' : AtomFeaturize(aprop, atomic_num_list), 'bond' : BondFeaturize(bprop), 'global' : GlobalFeaturize(gprop)}

def sm_to_mol(cann):
    return Chem.AddHs(Chem.MolFromSmiles(cann))

def single_predict(reac_sm, broken_bond_idx, prod_sms=None):
    r_m = sm_to_mol(reac_sm)

    # take from dataset bond splitting method
    Chem.Kekulize(r_m, clearAromaticFlags=True)
    bond = r_m.GetBondWithIdx(broken_bond_idx)
    mh = Chem.RWMol(r_m)
    a1 = bond.GetBeginAtomIdx()
    a2 = bond.GetEndAtomIdx()
    mh.RemoveBond(a1, a2)

    mh.GetAtomWithIdx(a1).SetNoImplicit(True)
    mh.GetAtomWithIdx(a2).SetNoImplicit(True)

    # Call SanitizeMol to update radicals
    Chem.SanitizeMol(mh)
    if prod_sms == None:
        p_ms = list(Chem.GetMolFrags(mh, asMols=True))
    else:
        p_ms = [sm_to_mol(sm) for sm in prod_sms]

    r_g = DGLwBDEMappings.dgl_from_mol(r_m)
    p_gs = [DGLwBDEMappings.dgl_from_mol(m) for m in p_ms]

    feats = {nt : [featurizers[nt](m) for m in [r_m, *p_ms]] for nt in featurizers}
    # feats = {nt : stders[nt].transform(torch.cat(feats[nt])) for nt in ['bond', 'atom', 'global']}
    feats = {nt : stders[nt](torch.cat(feats[nt])) for nt in ['bond', 'atom', 'global']}
    feats = {nt : torch.tensor(feats[nt], dtype=torch.float) for nt in ['bond', 'atom', 'global']}

    atom_map_for_rxn = prod_to_reac_atom_map([r_m, p_ms], [broken_bond_idx])
    bond_map_for_rxn = prod_to_reac_bond_map(p_ms, atom_map_for_rxn, r_m, broken_bond_idx)
    rxn_atom_mappings = DGLwBDEMappings.to_concat_map(atom_map_for_rxn)
    rxn_bond_mappings = DGLwBDEMappings.to_concat_map(bond_map_for_rxn)
    prods_has_bonds = [len(bm) > 0 for bm in bond_map_for_rxn[:-1]]

    rxn_feat_gen = BondDissociate(rxn_atom_mappings, rxn_bond_mappings, prods_has_bonds, None)
    rxn_feat_gen.reacs, rxn_feat_gen.prods = [0], [1, 2]

    pred = out_mean + (out_stdev * model(dgl.batch([r_g, *p_gs]), feats, [rxn_feat_gen]))

    return pred