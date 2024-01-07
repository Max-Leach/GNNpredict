from rdkit import Chem
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import subprocess, sys
import re

'''
Needs:
- [ ] Density distribution of pairwise bond lengths (load from xyz?)
- [ ] Functional group analysis (checkmol)
- [ ] Scaffold analysis
- [ ] self organising map analysis
- [X] frequency of fragmented bond types

remember to check read files fast with threading using python website saved in chemdata folder on chrome


also:

- add argv for format selection
- 

'''

#generate frequency of fragmented bond types from csv (keep in mind we may want to only load a csv file once)

def funcgroups():
    subprocess.run("for i in *.sdf;do checkmol -e #{i}; echo \; ;done > funcs.csv")

def bondfreq(filename, format):
    acf = pd.read_csv("./ALFABET_data/acp_updated.csv", names=["PID", "Parent", "Index","Child1", "Child2", "BDE", "Bond"])
    fig, ax = plt.subplots()
    acf['Bond'].value_counts().plot(ax=ax, kind='bar', xlabel='BondTypes',ylabel='Fequency')
    plt.savefig("bond_frequency." + format)

def logmoldistancesdf(mol, dfset):
    #count = 0
    #for every bond, compute distance, log it alongside bond-type
    mol = Chem.Mol(mol)
    #rdDetermineBonds.DetermineConnectivity(mol)
    mol = Chem.AddHs(mol)
    arr = mol.GetConformer().GetPositions()
    #AllChem.EmbedMolecule(mol)
    for bonds in mol.GetBonds():
        frag = mol.GetAtomWithIdx(bonds.GetBeginAtomIdx()).GetSymbol() + '-' + mol.GetAtomWithIdx(bonds.GetEndAtomIdx()).GetSymbol()
        print(str(bonds.GetIdx()) + ", " + str(frag))
        if frag not in dfset:
            dfset[frag] = pd.DataFrame(columns=['BondType', 'Distance'])
        
        dfset[frag].loc[len(dfset[frag])] = [frag] + [computedistance(bonds.GetBeginAtomIdx(), bonds.GetEndAtomIdx(), arr)]
        #df.loc[count] = [frag] + [computedistance(bonds.GetBeginAtomIdx(), bonds.GetEndAtomIdx(), arr)]
        #print(bonds.GetBeginAtomIdx(), bonds.GetEndAtomIdx())
        #count = count + 1

#make it so that it takes arguments, takes input files, creates a directory to output the figures
if __name__ == "__main__":
    bondfreq("./ALFABET_data/acp_updated.csv", "svg")