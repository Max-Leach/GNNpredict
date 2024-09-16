from novel_arch.deep_attn.data.featurizers import AtomFeaturize, BondFeaturize, GlobalFeaturize
from novel_arch.deep_attn.data.single_run_tools import prod_to_reac_atom_map, prod_to_reac_bond_map
from novel_arch.deep_attn.data.initial_containers import DGLwBDEMappings
from novel_arch.deep_attn.data.rxn_graph import BondDissociate
from rdkit import Chem
import pandas as pd
import numpy as np
import pickle
import torch
import dgl

atomic_num_list = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17] # for 1.6M dataset
# atomic_num_list = [1, 6, 7, 8]
aprop = ['atomic_num', 'total_degree', 'total_num_hs', 'ring_of_size', 'is_in_ring']
bprop = ['is_in_ring', 'ring_of_size', 'dative']
gprop = ['num_atoms', 'num_bonds', 'total_weight']
featurizers = {'atom' : AtomFeaturize(aprop, atomic_num_list), 'bond' : BondFeaturize(bprop), 'global' : GlobalFeaturize(gprop)}

with open('/home/moistry/Documents/research/data/1.6 results/1.6_scalers/16_output_scale', 'rb') as f:
    output_scale = pickle.load(f)
    out_mean, out_stdev = output_scale['mean'], output_scale['stdev']
with open('/home/moistry/Documents/research/data/1.6 results/1.6_scalers/16_stders', 'rb') as f:
    stders = pickle.load(f)
model_path = '/home/moistry/Documents/research/data/1.6 results/selected_trial-1-best'
model = torch.load(model_path, map_location='cpu')

dump_path = '/home/moistry/Documents/research/data/1.6 results/model_results.csv'

# Serial	Parentsmiles	BondIndex	Frag1smiles	Frag2smiles	BDH	BondType	Source

def sm_to_mol(cann):
    return Chem.AddHs(Chem.MolFromSmiles(cann))

csv_path = '/home/moistry/Documents/research/data/combined dset/combined_dataset COMPLETE.csv'
df = pd.read_csv(csv_path)
indices = tuple(range(len(df)))#[5:10]
df = df.loc(axis=0)[indices]

import time
start = time.perf_counter()

r_ms = [sm_to_mol(sm) for sm in df['Parentsmiles'].to_numpy()]
p1_ms = [sm_to_mol(sm) for sm in df['Frag1smiles'].to_numpy()]
p2_ms = [sm_to_mol(sm) for sm in df['Frag2smiles'].to_numpy()]
broken_bond_idxs = df['BondIndex'].to_numpy().tolist()

r_gs = [DGLwBDEMappings.dgl_from_mol(m) for m in r_ms]
p1_gs = [DGLwBDEMappings.dgl_from_mol(m) for m in p1_ms]
p2_gs = [DGLwBDEMappings.dgl_from_mol(m) for m in p2_ms]

feats = [{nt : [featurizers[nt](m) for m in [r_m, p1_m, p2_m]] for nt in featurizers} for r_m, p1_m, p2_m in zip(r_ms, p1_ms, p2_ms)]
feats = [{nt : stders[nt].transform(torch.cat(feat[nt])) for nt in ['bond', 'atom', 'global']} for feat in feats]
feats = [{nt : torch.tensor(feat[nt], dtype=torch.float) for nt in ['bond', 'atom', 'global']} for feat in feats]

# print(r_ms, p1_ms, p2_ms, broken_bond_idxs)
atom_map_for_rxns = [prod_to_reac_atom_map([r_m, [p1_m, p2_m]], [broken_bond_idx]) for r_m, p1_m, p2_m, broken_bond_idx in zip(r_ms, p1_ms, p2_ms, broken_bond_idxs)]
bond_map_for_rxns = [prod_to_reac_bond_map([p1_m, p2_m], atom_map_for_rxn, r_m, broken_bond_idx) for p1_m, p2_m, atom_map_for_rxn, r_m, broken_bond_idx in zip(p1_ms, p2_ms, atom_map_for_rxns, r_ms, broken_bond_idxs)]
rxn_atom_mappings = [DGLwBDEMappings.to_concat_map(atom_map_for_rxn) for atom_map_for_rxn in atom_map_for_rxns]
rxn_bond_mappings = [DGLwBDEMappings.to_concat_map(bond_map_for_rxn) for bond_map_for_rxn in bond_map_for_rxns]
prods_has_bonds = [[len(bm) > 0 for bm in bm_map[:-1]] for bm_map in bond_map_for_rxns]

rxn_feat_gens = [BondDissociate(rxn_atom_mapping, rxn_bond_mapping, prods_has_bond, None) for rxn_atom_mapping, rxn_bond_mapping, prods_has_bond in zip(rxn_atom_mappings, rxn_bond_mappings, prods_has_bonds)]
for rxn_feat_gen in rxn_feat_gens:
    rxn_feat_gen.reacs, rxn_feat_gen.prods = [0], [1, 2]

preds = [out_mean + (out_stdev * model(dgl.batch([r_g, p1_g, p2_g]), feat, [rxn_feat_gen])) for r_g, p1_g, p2_g, feat, rxn_feat_gen in zip(r_gs, p1_gs, p2_gs, feats, rxn_feat_gens)]
preds = [pred.detach().item() for pred in preds]

print('elapsed: {}'.format(time.perf_counter() - start))

tbl = np.array([indices, preds]).transpose()
res_df = pd.DataFrame(tbl.tolist(), columns=['Dataset Index', 'Model Inference'])
res_df.to_csv(dump_path, index=False)