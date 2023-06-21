from rdkit import Chem
import dgl
import networkx.algorithms.isomorphism as nx_iso
import networkx as nx
import torch

from novel_arch.deep_attn.data.initial_containers import DirectSmilesRepo, DGLwBDEMappings
from novel_arch.deep_attn.data.dataset import BDEDataset

dsr = DirectSmilesRepo()
dsr.append_reaction(['NCCCC(=O)O'], ['[CH2]CCC(=O)O', '[NH2]'], [0])
dsr.append_reaction(['NCCCC(=O)O'], ['[CH2]CC(=O)O', '[CH2]N'], [1])

dsr.append_reaction(['CN(N)CCCC(C)(C)C'], ['[CH2]CC(C)(C)C', '[CH2]N(C)N'], [3])
dsr.append_reaction(['COC(=O)OCCC#N'], ['CO[C]=O', 'N#CCC[O]'], [3])
dsr.append_reaction(['C/C=C/[C@@H](C)CCC#N'], ['[H]', 'C/C=C/[C@@H](C)[CH]CC#N'], [17])
# CN(N)CCCC(C)(C)C	3	[CH2]CC(C)(C)C	[CH2]N(C)N
# COC(=O)OCCC#N	3	CO[C]=O	N#CCC[O]
# C/C=C/[C@@H](C)CCC#N	17	[H]	C/C=C/[C@@H](C)[CH]CC#N
# CCC(=O)COC	4	C[O]	[CH2]C(=O)CC
# C#CCCC(=O)NCCO	2	C#C[CH2]	[CH2]C(=O)NCCO
# CC[C@@H](CC(C)=O)C(C)C	3	C[C]=O	[CH2][C@H](CC)C(C)C
# CCOc1cccc(O)c1	11	[H]	[CH2]COc1cccc(O)c1
# O=C[C@H](O)[C@H](O)[C@H](O)CO	8	[CH2][C@@H](O)[C@@H](O)[C@@H](O)C=O	[OH]


bdemap = DGLwBDEMappings(dsr)
from novel_arch.deep_attn.data.featurizers import AtomFeaturize, BondFeaturize, GlobalFeaturize
aprop = ['atomic_num', 'total_degree', 'total_num_hs', 'ring_of_size', 'is_in_ring']
bprop = ['is_in_ring', 'ring_of_size', 'dative']
gprop = ['num_atoms', 'num_bonds', 'total_weight']
dset = BDEDataset(dsr, bdemap, featurizers={'atom' : AtomFeaturize(aprop, [1, 6, 7, 8]), 'bond' : BondFeaturize(bprop), 'global' : GlobalFeaturize(gprop),}, load_graphs=True)
# print(dset[1])
# exit()
from novel_arch.deep_attn.data.dataloader import RxnDataLoader
loader = RxnDataLoader(dset, batch_size=2)
for graphs, feats, feat_gens, idxs in iter(loader):
    print(feats)
    print(len(dgl.unbatch(graphs)))
exit()
# print(bdemap.rxn_bond_mappings)
# print(bdemap.rxn_atom_mappings)
# # exit()
# mapping = bdemap.rxn_atom_mappings[1]
# # print(mapping)
# # print(len(mapping) == len(set(mapping)))
# print(bdemap.canon_to_dgl['NCCCC(=O)O'].nodes['atom'].data)
# prods = [bdemap.canon_to_dgl[c].nodes['atom'].data['specie'] for c in ['[CH2]CC(=O)O', '[CH2]N']]
# # print(torch.cat(prods))
# print(mapping)
# print(torch.all(torch.cat(prods)[mapping] == bdemap.canon_to_dgl['NCCCC(=O)O'].nodes['atom'].data['specie']))
# print([sorted(b) for b in bdemap.rxn_bond_mappings])
# # exit()
# i = 0
# mapping = bdemap.rxn_bond_mappings[i]
# # print(mapping)
# # print(len(mapping) == len(set(mapping)))
# print(bdemap.canon_to_dgl['NCCCC(=O)O'].nodes['bond'].data)
# prods = [bdemap.canon_to_dgl[c].nodes['bond'].data['species'] for c in ['[CH2]CCC(=O)O', '[NH2]']]
# prods.append(bdemap.canon_to_dgl['NCCCC(=O)O'].nodes['bond'].data['species'][dsr.react_broken_bonds[i][0]].unsqueeze(0))
# print('ahhhh', torch.cat(prods)[mapping])
# res = torch.cat(prods)[mapping]
# thing = bdemap.canon_to_dgl['NCCCC(=O)O'].nodes['bond'].data['species']
# for r, t in zip(res, thing):
#     print(set(r.tolist()) == set(t.tolist()))
# exit()

# CN(N)CCCC(C)(C)C	3	[CH2]CC(C)(C)C	[CH2]N(C)N
# COC(=O)OCCC#N	3	CO[C]=O	N#CCC[O]
# C/C=C/[C@@H](C)CCC#N	17	[H]	C/C=C/[C@@H](C)[CH]CC#N
# CCC(=O)COC	4	C[O]	[CH2]C(=O)CC
# C#CCCC(=O)NCCO	2	C#C[CH2]	[CH2]C(=O)NCCO
# CC[C@@H](CC(C)=O)C(C)C	3	C[C]=O	[CH2][C@H](CC)C(C)C
# CCOc1cccc(O)c1	11	[H]	[CH2]COc1cccc(O)c1

def mol_to_nx(mol):
    ng = nx.Graph()
    ng.add_nodes_from(range(mol.GetNumAtoms()))
    for b in mol.GetBonds():
        ng.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    for a in mol.GetAtoms():
        ng.nodes[a.GetIdx()]['specie'] = a.GetAtomicNum()
    return ng
r = (Chem.MolFromSmiles('CC[C@@H](CC(C)=O)C(C)C'))
r = Chem.AddHs(r)
ps = Chem.GetMolFrags(Chem.FragmentOnBonds(r, [3]))
mols = [(Chem.MolFromSmiles(s)) for s in ['C[C]=O', '[CH2][C@H](CC)C(C)C']]
mols = [Chem.AddHs(p) for p in mols]
# print(mol_to_nx(Chem.MolFromSmiles(sm)).nodes.data())
r = mol_to_nx(r)
ps = [r.subgraph(p) for p in ps]
print([p.edges for p in ps])
mols = [mol_to_nx(m) for m in mols]

nm = nx_iso.categorical_node_match('specie', 'BAD_MATCH')
gms = [nx_iso.GraphMatcher(m.to_undirected(), p.to_undirected(), nm) for p, m in zip(ps, reversed(mols))]
print([gm.is_isomorphic() for gm in gms])
print([gm.mapping for gm in gms])

exit()

# def sets_of_keys(sg):
#     return [set(sg_map.keys()) for sg_map in sg]
# r_refs = [sets_of_keys(sg) for sg in sg_maps]
# sg0, sg1 = r_refs
# joints = 0
# tot = 0
# for s0 in sg0:
#     for s1 in sg1:
#         tot += 1
#         if s0.isdisjoint(s1):
#             joints += 1
# print('number of joints', joints)
# print(len(sg0), len(sg1))
# print(r_refs[1])
# for i in range(2):
#     print("pioo", r_refs[0][i] == r_refs[0][i+1])

# for sg in gm.subgraph_isomorphisms_iter():
#     print(sg)
# print('total atom count', ps[p_i].number_of_nodes())

# mol = Chem.MolFromSmiles(sm)
# ng = nx.Graph()
# for b in mol.GetBonds():
#     ng.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
# print('edges wow eh', ng.edges)
# for a in mol.GetAtoms():
#     ng.nodes[a.GetIdx()]['specie'] = a.GetSymbol()
# print(ng.nodes.data())

# params = Chem.SmilesParserParams()
# params.removeHs = False
	# 12		C[CH]c1ccnc(C)n1 [H] 'CCc1ccnc(C)n1'

# big_daddy = (Chem.MolFromSmiles('CCc1ccnc(C)n1'))
# subset = (Chem.MolFromSmarts('C[CH]c1ccnc(C)n1'))
# matches = big_daddy.GetSubstructMatch(subset)
# print("wow you", matches)

sm = 'NCCCC(=O)O'
reac = Chem.MolFromSmiles(sm)
reac = Chem.AddHs(reac) # very important!

for b in reac.GetBonds():
    print(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    print(b.GetIdx())

print('number of fatoms', reac.GetNumAtoms())
for a in reac.GetAtoms():
    print(a.GetSymbol())
    print(a.GetAtomicNum())
    print(a.GetIdx())