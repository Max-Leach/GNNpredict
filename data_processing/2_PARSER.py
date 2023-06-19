from rdkit import Chem
from rdkit.Chem import PeriodicTable
import re

'''
Criteria includes:
 * <=15 Heavy atoms
 * Non-Isotopic
 * No formal charges
 * Limited to [C,H,N,O,P,S,Si,Cl,B,F] 
 * Building block* limited via bond types, and bonded atoms to selected cases[]
 * 3 and 4 membered rings involving cases[] removed

This program, conducts the criteria search and returns the valid molecules with an associated heavy atom count
'''

cases = {5,15,16,14}#Boron, Phosphorus, Sulfur, Silicon
bondstr = [['300'], ['300', '120', '310'], ['200', '210', '220'], ['400']]

heavy_max = 15
input_filename = 'minus-U_CID-SMILES'
output_filename = 'Processed_CID-SMILES'

def PARSE(file1, file2):
    count = 0
    for fline in iter(lambda: file1.readline(), ''): 
        count += 1
        if ((count % 1000000) == 0): print(count) #For visual inspection

        CID = re.search(r'\d+', fline).group()

        line = re.sub('^\d+', '', fline).strip() #ignore CID
        m = Chem.MolFromSmiles(line)
        if m == None: continue

        cond = SMILEProcess(m)
        if cond[0] and checkRing(m) and checkMol(m): file2.write(str(CID) + ' ' + cond[1] + ' ' + str(cond[2]) + ' ' + str(cond[3]) + '\n') #write to other file

#Only keep SMILES which satisfies our criteria
def SMILEProcess(m):
    h_count = 0
    default = [False, None, None, None]
    atomdiv = {1,6}
    div_num = 0

    for atom in m.GetAtoms():
        #Checking Heavy atom count, formal charge & atomic number
        AN = atom.GetAtomicNum()
        AM = atom.GetIsotope() 
        h_count +=1
        
        #Add atomic diversity info
        if AN not in atomdiv: 
            atomdiv.add(AN)
            div_num += AN

        if is_radical() or is_too_heavy() or is_charged() or is_isotope() or is_ext_element():
            return default
    return True, Chem.MolToSmiles(m), h_count, div_num

def is_radical(atom):
    if (atom.GetNumRadicalElectrons() != 0): return True
    return False

def is_too_heavy(h_count):
    if (h_count > heavy_max): return True
    return False

def is_charged(atom):
    if (atom.GetFormalCharge() != 0): return True
    False

def is_isotope(AM):
    if (AM != 0): return True
    return False

def is_ext_element(AN):
    if ((AN == 2) | (AN == 3) | (AN == 4) | (AN == 10) | (AN == 11) | (AN == 12) | (AN == 13) | (AN >= 18)): return True
    return False

def checkMol(molecule):
    #molecule = Chem.MolFromSmiles(moleculestr)
    molecule = Chem.AddHs(molecule)
    
    for i in molecule.GetAtoms():
        singles = 0
        doubles = 0
        triples = 0
        AtomNo = i.GetAtomicNum()

        index = cases.index(AtomNo) if AtomNo in cases else -1
        if (index == -1): continue
        for j in i.GetBonds():
            bondtype = j.GetBondType()

            if bondtype == 1.0:
                singles += 1
            if bondtype == 2.0:
                doubles += 1
                if j.GetEndAtom().GetAtomicNum() != 8 and  j.GetBeginAtom().GetAtomicNum() != 8:
                    return False
            if bondtype == 3.0:
                triples += 1
        bonds = '' + str(singles) + str(doubles) + str(triples) #aggregates bond information in the form '{singles}{doubles}{triples}'
        if bonds not in bondstr[index]:
            return False
    return True 

def checkRing(molecule):
    for atom in molecule.GetAtoms():
        if  (atom.GetAtomicNum() in cases) and (atom.IsInRingSize(3) or atom.IsInRingSize(4)):
                return False
    return True

if __name__ == '__main__':
    with open(input_filename, 'r') as fl1, open(output_filename, 'a') as fl2:
        PARSE(fl1,fl2)
