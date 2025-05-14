import argparse
import logging
import torch
import pickle
import json
from util import predict_all

# run inference on input reaction

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    torch.multiprocessing.set_forkserver_preload(["torch"])

    parser = argparse.ArgumentParser(description='Inference on all valid bonds on single reactant, expects model file to be in parent directory')
    parser.add_argument("reactant_smiles", type=str, help="smiles string for reactant to undergo bond separation")

    args = parser.parse_args()

    bond_idxs, preds = predict_all(args.reactant_smiles)

    print('RDkit bond indices:')
    print(bond_idxs)
    print('Predictions:')
    print(preds.numpy())