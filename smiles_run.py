import pandas as pd
from architecture.data.rxn_graph import BondDissociate
from architecture.data.initial_containers import DGLwBDEMappings
from architecture.data.single_run_tools import prod_to_reac_atom_map, prod_to_reac_bond_map
from architecture.data.featurizers import AtomFeaturize, BondFeaturize, GlobalFeaturize
from rdkit import Chem
import torch
import dgl
import pickle
import time
import numpy as np
import pandas as pd

reac_sm = 'CC[C@H]1C[C@H](OS(C)(=O)=O)CO1'
prod_sms = ['CC[C@H]1C[C@H]([O])CO1', 'C[S](=O)=O']
broken_bond_idx = 5

with open('inference/12_transforms', 'rb') as f:
    transforms = pickle.load(f)

out_mean, out_stdev = transforms['val_mean'], transforms['val_stdev']
stders = transforms['transform']

model_path = 'inference/best_model'
model = torch.load(model_path, map_location='cpu')

df = pd.read_csv("C:/Users/Maxim/Documents/DeepBDE/unified_02-11_no_extras.csv")

# atomic_num_list = [1, 6, 7, 8]
atomic_num_list = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17]
aprop = ['atomic_num', 'total_degree', 'total_num_hs', 'ring_of_size', 'is_in_ring']
bprop = ['is_in_ring', 'ring_of_size', 'dative']
gprop = ['num_atoms', 'num_bonds', 'total_weight']
featurizers = {'atom' : AtomFeaturize(aprop, atomic_num_list), 'bond' : BondFeaturize(bprop), 'global' : GlobalFeaturize(gprop)}

def sm_to_mol(cann):
    return Chem.AddHs(Chem.MolFromSmiles(cann))

start = time.perf_counter()

r_m = sm_to_mol(reac_sm)
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
elapsed = time.perf_counter() - start
print(pred)
print('elapsed: {}'.format(elapsed))


def sm_to_atomic_num(mol):
    return [atom.GetAtomicNum() for atom in mol.GetAtoms()]

#reac_sm = 'CC[C@H]1C[C@H](OS(C)(=O)=O)CO1'
#prod_sms = ['CC[C@H]1C[C@H]([O])CO1', 'C[S](=O)=O']
#broken_bond_idx = 5

captured_input = None

def forward_hook(module, input, output):
    global captured_input
    captured_input = input
    #print("Input: ", input)
    hook_handle.remove()

rows = np.array([])
t1 = time.time()
for i in range(1):
    row_ = df.iloc[i]
    reac_sm = row_['Parentsmiles']
    prod_sms = [row_['Frag1smiles'], row_['Frag2smiles']]
    bond_type = row_['BondType']
    broken_bond_idx = int(row_['BondIndex'])

    r_m = sm_to_mol(reac_sm)
    p_ms = [sm_to_mol(sm) for sm in prod_sms]

    #atomic_num_list = sm_to_atomic_num(r_m)
    #print(atomic_num_list)

    aprop = ['atomic_num', 'total_degree', 'total_num_hs', 'ring_of_size', 'is_in_ring']
    bprop = ['is_in_ring', 'ring_of_size', 'dative']
    gprop = ['num_atoms', 'num_bonds', 'total_weight']
    featurizers = {'atom' : AtomFeaturize(aprop, atomic_num_list), 'bond' : BondFeaturize(bprop), 'global' : GlobalFeaturize(gprop)}



    start = time.perf_counter()


    #reactant_no_bonds = r_m.GetNumBonds()
    #product_no_bonds = [m.GetNumBonds() for m in p_ms]
    #print("No. of bonds in reactant: ", reactant_no_bonds)
    #print("No. of bonds in products: ", product_no_bonds)
    #print("Total no. of bonds in reaction: ", reactant_no_bonds + sum(product_no_bonds))

    r_g = DGLwBDEMappings.dgl_from_mol(r_m)
    p_gs = [DGLwBDEMappings.dgl_from_mol(m) for m in p_ms]

    feats = {nt : [featurizers[nt](m) for m in [r_m, *p_ms]] for nt in featurizers}
    #feats = {nt : stders[nt].transform(torch.cat(feats[nt])) for nt in ['bond', 'atom', 'global']}
    feats = {nt: stders[nt](torch.cat(feats[nt])) for nt in ['bond', 'atom', 'global']}

    feats = {nt : torch.tensor(feats[nt], dtype=torch.float) for nt in ['bond', 'atom', 'global']}

    atom_map_for_rxn = prod_to_reac_atom_map([r_m, p_ms], [broken_bond_idx])
    bond_map_for_rxn = prod_to_reac_bond_map(p_ms, atom_map_for_rxn, r_m, broken_bond_idx)
    rxn_atom_mappings = DGLwBDEMappings.to_concat_map(atom_map_for_rxn)
    rxn_bond_mappings = DGLwBDEMappings.to_concat_map(bond_map_for_rxn)
    prods_has_bonds = [len(bm) > 0 for bm in bond_map_for_rxn[:-1]]

    rxn_feat_gen = BondDissociate(rxn_atom_mappings, rxn_bond_mappings, prods_has_bonds, None)
    rxn_feat_gen.reacs, rxn_feat_gen.prods = [0], [1, 2]

    #print(model)


    #hook_handle.remove()


    # Register the hook
    hook_handle = model.fc_to_scalar[0].register_forward_hook(forward_hook)
    #torch.cuda.synchronize() if torch.cuda.is_available() else None

    #print("Captured input:", captured_input)

    #def forward_hook(module, input, output):
        #print(f"Inside forward hook for {module.__class__.__name__}")
     #   print("Input: ", input)

        #return input
        #print(input[0].shape)
        #print("--------")
        #print(output)
        #print(output.shape)
        #print("--------")
        #print(output['bond'])
        #print(output['bond'].shape)

        #print(f"Input shape: {input.keys[0].shape}")
        #print(f"Output shape: {output.shape}")
        #print("--------")

    #print("#################################################################")

    #hook_handle = model.graph_net[0].register_forward_hook(forward_hook)
    #hook_handle = model.fc_to_scalar[0].register_forward_hook(forward_hook)



    pred = out_mean + (out_stdev * model(dgl.batch([r_g, *p_gs]), feats, [rxn_feat_gen]))
    #elapsed = time.perf_counter() - start
    #pred_scalar = detached_tensor = pred.detach().item()

    #print(pred_scalar)
    #print('elapsed: {}'.format(elapsed), "\n")

    captured_input_element = captured_input[0]
    captured_input_detached = captured_input_element.detach().numpy().tolist()

    new_row = {
        'Parentsmiles': reac_sm,
        'Frag1smiles': prod_sms[0],
        'Frag2smiles': prod_sms[1],
        'Bondindex': broken_bond_idx,
        'BondType': bond_type,
        'BDE': df['BDH'][i],
        #'Predicted_BDE': pred_scalar,
        'Embeddings' : captured_input_detached
    }
    rows = np.append(rows, new_row)


cols = new_row.keys()
df = pd.concat([pd.DataFrame([row], columns=cols) for row in rows],
          ignore_index=True)

df.to_csv("C:/Users/Maxim/Documents/DeepBDE/embeddings/out_2.csv")
##
# Notes:
# Check if its actually fc_graph_input -> DONE
# Flatten tensor
# Dim reduction
# might need to batch process dataframes and then combine at the end
# save dataframe with parentsmiles, frag1smiles, frag2smiles, bde, bondindex, and then the embeddings
# the captured input seems to be one index behind so we must to a final captured input for the last one
#   and ensure that the indexs align for the correct SMILES row
##
t2 = time.time()
print("Time taken: ", t2-t1, " seconds")