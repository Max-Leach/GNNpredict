from rdkit import Chem
import glob
import sys
import shutil
import networkx as nx
'''
Outputs inch_diff file containing information on matches/non-matches with original files

reads from molecule_list and outputs inch_diff with results

checks structural isomorphism of MolFile with SmilesToMol object

'''

def topology_from_rdkit(rdkit_molecule):

    topology = nx.Graph()
    for atom in rdkit_molecule.GetAtoms():
        # Add the atoms as nodes
        topology.add_node(atom.GetIdx())

        # Add the bonds as edges
        for bonded in atom.GetNeighbors():
            topology.add_edge(atom.GetIdx(), bonded.GetIdx())

    return topology


def is_isomorphic(topology1, topology2):
    return nx.is_isomorphic(topology1, topology2)


with open('molecule_list', 'r') as file1, open('inch_diff', 'w') as out1, open('struc_diff', 'w') as out2:
    for line in iter(lambda: file1.readline(), ''):
        split = line.split(' ')
        hit = glob.glob('sdfs/' + split[0] + '_*.sdf')
        if hit:
            for file in hit: filename = file
        else:
            #out.writelines('Incorrect\n')
            continue
        try:
            m1 = Chem.MolFromSmiles(split[1], sanitize=False)
            m2 = Chem.MolFromMolFile(filename)

            #rdDetermineBonds.DetermineConnectivity(m2)
            #Chem.SanitizeMol(m2)

            inchi1 = Chem.MolToInchi(m1, options='/DoNotAddH').split('/')
            inchi2 = Chem.MolToInchi(m2, options='/DoNotAddH').split('/')

            inchi1 = inchi1[0] + '/' +inchi1[1] + '/' +inchi1[2]
            inchi2 = inchi2[0] + '/' +inchi2[1] + '/' +inchi2[2]

            top1 = topology_from_rdkit(m1)
            top2 = topology_from_rdkit(m2)

            if (not is_isomorphic(top1, top2)):
                out2.writelines(split[0] + ' Smile: ' + inchi1 + ' File: ' + inchi2)

            #fails too oftern to be reliable as of right now
            #if (not m1.HasSubstructMatch(m2)):
            #    out2.writelines(split[0] + ' Smile: ' + inchi1 + ' File: ' + inchi2 + ' True' +'\n')

            if (inchi1 == inchi2):
                out1.writelines(split[0] + ' Smile: ' + inchi1 + ' File: ' + inchi2 + ' True' +'\n')
            else:
                out1.writelines(split[0] + ' Smile: ' + inchi1 + ' File: ' + inchi2 + ' False' +'\n')
                #shutil.move(filename, 'incorrect/' + filename)
        except:
                out1.writelines(line.rstrip('\n') + ' RDKIT incompatible\n')