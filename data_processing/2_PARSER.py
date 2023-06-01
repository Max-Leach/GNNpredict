from rdkit import Chem
from rdkit.Chem import PeriodicTable
import re

pt = Chem.GetPeriodicTable()

'''
Criteria includes:
 * <=15 Heavy atoms
 * Non-Isotopic
 * No formal charges
 * Limited to [C,H,N,O,P,S,Si,Cl,B,F] 
'''

heavy_max = 15
input_filename = 'minus-U_CID-SMILES'
output_filename = 'Processed_CID-SMILES'

def PARSE(file1, file2):
    count = 0
    for fline in iter(lambda: file1.readline(), ''): 
        count += 1
        if ((count % 1000000) == 0): print(count) #For visual inspection

        CID = re.search(r'\d+', fline).group()
        cond = SMILEProcess(fline)
        if cond[0]: file2.write(str(CID) + ' ' + cond[1] + ' ' + str(cond[2]) +'\n') #write to other file

#Only keep SMILES which satisfies our criteria
def SMILEProcess(line):
    h_count = 0
    line = re.sub('^\d+', '', line).strip() #ignore CID
    default = [False, None, 0]

    m = Chem.MolFromSmiles(line) #Use rdkit to built molecule
    if m == None: return default

    for atom in m.GetAtoms():
        #Checking Heavy atom count, formal charge & atomic number
        AN = atom.GetAtomicNum()
        AM = atom.GetIsotope() 
        h_count +=1
        
        if (h_count > heavy_max): return default
        if (atom.GetFormalCharge() != 0): return default
        if (AM != 0) and (AM != PeriodicTable.GetMostCommonIsotope(pt, AN)): return default
        if ((AN == 2) | (AN == 3) | (AN == 4) | (AN == 10) | (AN == 11) | (AN == 12) | (AN == 13) | (AN >= 18)): return default
    return True, Chem.MolToSmiles(m), h_count

if __name__ == '__main__':
    with open(input_filename, 'r') as fl1, open(output_filename, 'a') as fl2:
        PARSE(fl1,fl2)
