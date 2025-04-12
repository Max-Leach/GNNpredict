from smiles_parent_only_run import predict
import pandas as pd
import pickle
import argparse

# NOTE: update for public use

parser = argparse.ArgumentParser("which parent number to run inference on")
parser.add_argument("parent_no", type=int)
args = parser.parse_args()

# parent_no = 1
dir_path = '/home/moistry/Documents/research/data/applicatino/results'
smi_df = pd.read_csv(dir_path + '/parent_list.csv')
parent_smi = smi_df['Parent Smiles'].iloc[args.parent_no - 1]
bond_df = pd.read_csv(dir_path + '/parent_{}.csv'.format(args.parent_no))
bond_idxs = bond_df['Bond Index']

print('parent info:', parent_smi)
print(bond_idxs)

sm_to_bond_idxs = {
    parent_smi : bond_idxs,
}

# {
    # 'COc1ccc(NC=O)c(C(=O)CCNC(C)=O)c1' : [
    #     0,
    #     19,
    #     24,
    #     25,
    #     26,
    #     27,
    #     28,
    #     29,
    #     30,
    #     31,
    #     32,
    #     33,
    # ],

    # 'O=C(O)/C=C/c1ccc(O)c(O)c1' : [
    #     19,
    #     18,
    # ],

    # 'COc1ccc(N)c(C(=O)CCNC(C)=O)c1' : [
    #     19,
    #     5,
    #     24,
    #     27,
    #     28,
    #     29,
    # ]
# }

pairs = [(sm, ix) for sm, idxs in sm_to_bond_idxs.items() for ix in idxs]
# print(pairs)
preds = [predict(*args) for args in pairs]
bond_df['DeepBDE'] = [p.detach().item() for p in preds]

bond_df.to_csv(dir_path + '/parent_{}.csv'.format(args.parent_no), index=False)