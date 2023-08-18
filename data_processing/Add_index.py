from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
import rdkit
import pandas as pd
import re

''' Added bond indices, redundant as is present in 4_fragmentgen.py'''

def ld_points(line):
    split = line.split(',')
    return [split[2], split[3], split[4]]

def determine_index(parent, frag1, frag2):
    '''print(parent, end=' ')
    print(frag1, end=' ')
    print(frag2)'''
    
    parent_mol  = Chem.MolFromSmiles(parent)
    frag_mol1 = Chem.MolFromSmiles(frag1, sanitize=False)
    frag_mol2 = Chem.MolFromSmiles(frag2, sanitize=False)

    match1 = list(parent_mol.GetSubstructMatch(frag_mol1))
    match2 = list(parent_mol.GetSubstructMatch(frag_mol2))

    try:
        allbonds = find_bonds(parent_mol, frag_mol1, match1) + find_bonds(parent_mol,frag_mol2,match2)
        allbonds.sort()
    except:
        return None
    return int(find_index(allbonds))

def find_bonds(parent, frag, match):
    hit_bonds = []
    for bond in frag.GetBonds():
        aid1 = match[bond.GetBeginAtomIdx()]
        aid2 = match[bond.GetEndAtomIdx()]
        hit_bonds.append(parent.GetBondBetweenAtoms(aid1,aid2).GetIdx())
    return hit_bonds

def find_index(bond_list):
    pointer = 0
    for i in range(len(bond_list) - 1):
        if (bond_list[i] + 1 != bond_list[i+1]):
            pointer = bond_list[i] + 1
            break
        pointer = bond_list[i+1] + 1
    return pointer

if __name__ == '__main__':
    df = pd.read_csv('selection.csv')
    df.insert(6,"BDH", None)
    df.insert(7, "BondIndex", None)
    
    df['BondIndex'] = df.apply(lambda row: determine_index(row['Pid'], row['Frag1'], row['Frag2']), axis=1)
    df.to_csv('subset_dataset.csv', index=False)
