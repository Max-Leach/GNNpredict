from rdkit import Chem
import sys
import shutil
import os

'''
Compares input and output sdf files and moves non-matches to incorrect directory

used as so: compare.py input_file.sdf output_file.sdf file.log log.out 

'''

with open(sys.argv[4], 'a') as f:
    if not os.path.exists('molecules/incorrect'): os.makedirs('molecules/incorrect')

    line = sys.argv[1]  + '  '
    try:
        inchi1 = Chem.MolToInchi(Chem.MolFromMolFile(sys.argv[1])).split('/')
        inchi2 = Chem.MolToInchi(Chem.MolFromMolFile(sys.argv[2])).split('/')
        
        inchi1 = inchi1[0] + '/' +inchi1[1] + '/' +inchi1[2]
        inchi2 = inchi2[0] + '/' +inchi2[1] + '/' +inchi2[2]

        similar = (inchi1==inchi2)

        line += inchi1 + '  ' + inchi2 +'  ' + str(similar)
        f.writelines(line + '\n')
        #if (not similar):
            #for i in range(1,4): shutil.move(sys.argv[i], 'incorrect/' + sys.argv[i])
    except:
        with open('errors.out', 'a') as f2:
            f2.writelines(sys.argv[1] + '  None\n')
            #for i in range(1,4): shutil.move(sys.argv[i], 'incorrect/' + sys.argv[i])
