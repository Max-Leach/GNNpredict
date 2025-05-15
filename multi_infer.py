import argparse
import logging
import torch
import pickle
import json
from architecture.inference_util import multi_predict

# run inference on input reaction

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    torch.multiprocessing.set_forkserver_preload(["torch"])

    parser = argparse.ArgumentParser(description='Multiple reaction inference on single reactant, expects model file to be in parent directory')
    parser.add_argument("reactant_smiles", type=str, help="smiles string for reactant to undergo bond separation")
    parser.add_argument("rdkit_bond_idxs", type=json.loads, help="list of indices of bond of reactant_smiles for bond separation, bond indices assigned by RDkit")

    args = parser.parse_args()

    print(multi_predict(args.reactant_smiles, args.rdkit_bond_idxs).numpy())