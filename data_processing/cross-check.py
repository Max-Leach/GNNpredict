from rdkit import Chem
import glob

'''
Outputs inch_diff file containing information on matches/non-matches with original files

reads from molecule_list and outputs inch_diff with results

'''

with open('molecule_list', 'r') as file1, open('inch_diff', 'a') as out:
    for line in iter(lambda: file1.readline(), ''): 
        split = line.split(' ')
        hit = glob.glob(split[0] + '_*_o.sdf') 
        if hit: 
            for file in hit: filename = file
        else:
            #out.writelines('Incorrect\n')
            continue
        try:
            m1 = Chem.MolFromSmiles(split[1], sanitize=False)
            m2 = Chem.MolFromMolFile(filename)

            inchi1 = Chem.MolToInchi(m1, options='/DoNotAddH').split('/')
            inchi2 = Chem.MolToInchi(m2, options='/DoNotAddH').split('/')
           
            inchi1 = inchi1[0] + '/' +inchi1[1] + '/' +inchi1[2]
            inchi2 = inchi2[0] + '/' +inchi2[1] + '/' +inchi2[2]

            if (inchi1 == inchi2):
                out.writelines(split[0] + ' Smile: ' + inchi1 + ' File: ' + inchi2 + ' True' +'\n')
                #out.writelines(filename + ' ' + hit[0] + '\n')
            else:
                out.writelines(split[0] + ' Smile: ' + inchi1 + ' File: ' + inchi2 + ' False' +'\n')
        except:
                out.writelines(line.rstrip('\n') + ' RDKIT incompatible\n')
