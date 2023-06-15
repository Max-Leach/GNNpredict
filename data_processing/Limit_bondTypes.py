from rdkit import Chem
import sys

'''
[3]B -> 3 single bonds 

[3]P -> 3 single bonds
[2]P -> 2 double bonds, 1 single bond
[4]P -> 3 single bonds, 1 double bond

[2]S -> 2 single bonds
[3]S -> 2 single bonds, 1 double bond
[4]S -> 2 single bonds, 2 double bonds

[4]Si -> 4 single bonds


Step 1: iterate through bonds
step 2: use GetOtherAtomidx on the bond class and check to ensure it's legal
'''
cases = [5,15,16,14]#Boron, Phosphorus, Sulfur, Silicon
bondstr = [['300'], ['300', '120', '310'], ['200', '210', '220'], ['400']]

def checkMol(moleculestr):
    molecule = Chem.MolFromSmiles(moleculestr)
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
                if (AtomNo == 5):
                    return False
                if j.GetEndAtom().GetAtomicNum() != 8 and  j.GetBeginAtom().GetAtomicNum() != 8:
                    return False
            if bondtype == 3.0:
                triples += 1
        bonds = '' + str(singles) + str(doubles) + str(triples) #aggregates bond information in the form '{singles}{doubles}{triples}'
        if bonds not in bondstr[index]:
            return False
    return True 

def loadparent(line):
    return line.split(',')[2]

if __name__ == '__main__':
    with open(sys.argv[1], 'r') as f1, open(sys.argv[2], 'w') as f2:
        for line in iter(lambda: f1.readline(), ''): 
            if (checkMol(loadparent(line))):
                f2.write(line)
