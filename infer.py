import argparse
import logging
import torch
import pickle
from architecture.inference_util import single_predict

# run inference on input reaction

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    torch.multiprocessing.set_forkserver_preload(["torch"])

    parser = argparse.ArgumentParser(description='Single reaction inference, expects model file to be in parent directory')
    parser.add_argument("reactant_smiles", type=str, help="smiles string for reactant to undergo bond separation")
    parser.add_argument("rdkit_bond_idx", type=int, help="index of bond of reactant_smiles for bond separation, bond indices assigned by RDkit")
    parser.add_argument("--product_1_smiles", type=str, required=False, help="smiles of one of the products, can be swapped with 2. only useful if you want to check that bond index corresponds with reaction you expect. if you get an error, it means the bond separation products does not match what is given.")
    parser.add_argument("--product_2_smiles", type=str, required=False, help="smiles of one of the products, can be swapped with 1. only useful if you want to check that bond index corresponds with reaction you expect. if you get an error, it means the bond separation products does not match what is given.")

    args = parser.parse_args()

    prod_sms = [args.product_1_smiles, args.product_2_smiles]
    if any([p != None for p in prod_sms]):
        if not all([p != None for p in prod_sms]):
            print("You either need both product smiles or none!")
            exit()
    else:
        prod_sms = None

    print(single_predict(args.reactant_smiles, args.rdkit_bond_idx, prod_sms).numpy())