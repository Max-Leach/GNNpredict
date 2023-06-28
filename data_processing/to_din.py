
def Addinfo(file, index, ind_line, split, bond1, bond2):
    file.write('1\n' + ind_line[3] + '_' + bond1 + '_2.xyz\n')
    file.write('1\n' + ind_line[4] + '_' + bond2 + '_2.xyz\n')
    file.write('-1\n'+ ind_line[2] + '_' + split[8] + '_1.xyz\n')
    file.write('0\n' + split[index]+ ' ' + split[0] + '_' + split[8] + '\n')

def find_bond(mols, index):
    for i in mols: 
        if i.split(' ')[0] == index:
            return i.split(' ')[4].rstrip('\n')

with open('indexed_selection', 'r') as ind, open('updated_dataset.csv', 'r') as inp, open('dataset_bdenergy.din', 'w') as o1, open('dataset_bdenthalpy.din', 'w') as o2, open('molecule_list', 'r') as o3:
    inp.readline()
    ind.readline()
    mols = o3.readlines()
    for line in iter(lambda: inp.readline(), ''):
        split = line.split(',')
        print(split)
        for line2 in iter(lambda: ind.readline(), ''):
            ind_line = line2.split(',')
            bond1 = find_bond(mols, ind_line[3])
            bond2 = find_bond(mols, ind_line[4])
            
            if (split[0] == ind_line[0]):
               Addinfo(o1, 5, ind_line, split, bond1, bond2)
               Addinfo(o2, 6, ind_line, split, bond2, bond2)
               break
            else:
               continue
