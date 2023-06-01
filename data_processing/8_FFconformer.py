import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToXYZFile
import os
import time
import sys

def optimize_molecule_UFF(SMILES):
        """ Embed a molecule in 3D space, optimizing a number of conformers and
        selecting the most stable
        """

        mol = Chem.MolFromSmiles(SMILES)
        mol = Chem.rdmolops.AddHs(mol)

        # If the molecule is a radical; add a hydrogen so MMFF converges to a
        # reasonable structure
        is_radical = False
        radical_index = None
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetNumRadicalElectrons() != 0:
                is_radical = True
                radical_index = i

                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
                atom.SetNumRadicalElectrons(0)

        # Use min < 3^n < max conformers, where n is the number of rotatable bonds
        NumRotatableBonds = AllChem.CalcNumRotatableBonds(mol)
        NumConformers = np.clip(
            3 ** NumRotatableBonds, 100, 1000
        )

        conformers = AllChem.EmbedMultipleConfs(
            mol,
            numConfs=int(NumConformers),
            pruneRmsThresh=0.2,
            randomSeed=1,
            useExpTorsionAnglePrefs=True,
            useBasicKnowledge=True,
        )

        if (conformers):
            try:
                most_stable_conformer = RDKITConformer(mol, conformers)

                # If hydrogen was added; remove it before returning the final mol
                if is_radical:isRadical(mol, radical_index)
                return mol, int(most_stable_conformer)
            except:
                if is_radical:isRadical(mol, radical_index)
                return mol, None

            #write to xyz file
        else:
            if is_radical:isRadical(mol, radical_index)
            return mol, None


        return mol, int(most_stable_conformer)

def RDKITConformer(mol, conformers):
    Chem.SanitizeMol(mol)

    def optimize_conformer(conformer):
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conformer)
                ff.Minimize()
                return float(ff.CalcEnergy())

    if len(conformers) == 1:
            most_stable_conformer = conformers[0]
    else:
        conformer_energies = np.array(
            [optimize_conformer(conformer) for conformer in conformers]
        )

        most_stable_conformer = conformer_energies.argmin()
    return int(most_stable_conformer)

def isRadical(mol, radical_index):
    radical_atom = mol.GetAtomWithIdx(radical_index)
    radical_atom.SetNumExplicitHs(int(radical_atom.GetNumExplicitHs()) - 1)
    radical_atom.SetNumRadicalElectrons(1) 


def RDKITwritetoXYZ(Id, arg, script_dir, SDFdirectory):
    MolToXYZFile(mol, Id[0] + '.xyz', confId=arg ) #move this to other directory
    with open(Id[0] + '.xyz', 'r') as rfile:
        lines = rfile.readlines()
        lines[1] = Id[2] + ' ' + Id[3].rstrip('\n') + ' ' + Id[1] + '\n'

    with open(os.path.join(script_dir, SDFdirectory + '/' + Id[0]+ '_' + Id[4].rstrip('\n')+ '_' + Id[3] + '.xyz'), 'w') as wfile:
        wfile.writelines(lines)
    os.remove(Id[0] + '.xyz') 

if __name__ == "__main__":
    SDFdirectory = 'molecules'
    count = 0
    list_file = sys.argv[1] 
    err_file = sys.argv[1] 
    if not os.path.exists(SDFdirectory): os.makedirs(SDFdirectory) #Create directory iff doesn't exist
    start = time.time()
    print("Starting FF optimizations")

    with open(list_file, 'r') as listfile, open(err_file, 'w'):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.split(script_path)[0] #find the absolute filepath to current directory

        for line in iter(lambda: listfile.readline(), ''): 
            Id = line.split(' ') #split into [Index, molecule]
            count += 1

            try:
                mol, arg = optimize_molecule_UFF(Id[1])
            except:
                err_file.writelines(line)
                continue
            
            if arg != None: RDKITwritetoXYZ(Id,arg,script_dir,SDFdirectory )
            else:
                Chem.MolToMolFile(mol, Id[0] + '.sdf')
                os.system("obabel " + Id[0] + ".sdf -O " + Id[0] + "_opt.sdf --minimize --steps 1500 --sd --ff UFF")
                os.system("obabel -i sdf " + Id[0]+"_opt.sdf -o xyz -O " +Id[0] +".xyz")
                os.remove(Id[0] + '.sdf')
                os.remove(Id[0] + "_opt.sdf")
                
                with open(Id[0] + '.xyz', 'r') as rfile:
                    lines = rfile.readlines()
                    lines[1] = Id[2] + ' ' + Id[3].rstrip('\n') + ' ' + Id[1] + '\n'

                with open(os.path.join(script_dir, SDFdirectory + '/' + Id[0]+ '_' + Id[4].rstrip('\n')+ '_' + Id[3] + '.xyz'), 'w') as wfile:
                    wfile.writelines(lines)
                    os.remove(Id[0] + '.xyz') 
            print(count)
    print("Optimizations complete, time elapsed: " + str(time.time()-start))
