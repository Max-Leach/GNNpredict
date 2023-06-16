from rdkit import Chem

from novel_arch.deep_attn.data import DirectSmilesRepo, DGLwRxnMappings

# poo = DirectSmilesRepo()
# poo.append_reaction(['NCCCC(=O)O'], ['[CH2]CCC(=O)O', '[NH2]'])
# poo.append_reaction(['NCCCC(=O)O'], ['[CH2]CC(=O)O', '[CH2]N'])
# print(len(poo.canon_to_mol))
# print(len(poo.r_p_canon))

sm = '[CH2]N'
mol = Chem.MolFromSmiles(sm)
graff = DGLwRxnMappings.dgl_from_mol(mol)
print(graff)

assert False, "boohoo"

sm = 'NCCCC(=O)O'
reac = Chem.MolFromSmiles(sm)
reac = Chem.AddHs(reac) # very important!

for b in reac.GetBonds():
    print(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    print(b.GetIdx())

print('number of fatoms', reac.GetNumAtoms())
for a in reac.GetAtoms():
    print('pee')
    print(a.GetSymbol())
    print(a.GetAtomicNum())
    print(a.GetIdx())