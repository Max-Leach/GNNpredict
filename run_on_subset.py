### runs on subset of dataset given dictionary of bond types -> indices

from novel_arch.deep_attn.data.dataloader import RxnDataLoader
from novel_arch.deep_attn.data.dataset import BDEDataset, BDESubset

import pickle as pik
import torch
import pandas as pd

subset_path = '/home/moistry/Documents/research/data/286_indices/test_indices_d'
dump_path = '/home/moistry/Documents/research/data/286 results/model_results.csv'
trial_path = '/home/moistry/Documents/research/data/286 results/trial_286_retrain.sub'
dset_path = '/home/moistry/Documents/research/data/old_stuff/dset_lazy/dset'

model_path = trial_path + '/best_model'

dset = BDEDataset.load(dset_path)
dset.load_graphs = True
with open(subset_path, 'rb') as valid_indices_f:
    valid_indices_d = pik.load(valid_indices_f)

model = torch.load(model_path, map_location='cpu')

print('Dataset size:', len(dset))
print('Valid subset sizes:', {bt : len(idxs) for bt, idxs in valid_indices_d.items()})

def test_on_idxs(idxs):
    test_batch_size = 512
    ldr = RxnDataLoader(BDESubset(dset, idxs), batch_size=test_batch_size, num_workers=4)
    model.eval()
    nested_table = [[ts, (model(gs, fs, rxns) * dset.val_stdev + dset.val_mean).detach().squeeze(), torch.tensor(sub_idxs)] for (gs, fs, rxns, sub_idxs), ts in iter(ldr)]
    table = [[item[i] for item in nested_table] for i in range(len(nested_table[0]))]
    table = [torch.cat(col) for col in table]
    return torch.stack([torch.tensor(idxs)] + table).transpose(0, 1)

def df_test_on_idxs(idxs):
    # create pandas dataframe over given idxs
    tbl = test_on_idxs(idxs)
    df = pd.DataFrame(tbl, columns=['Dataset Index', 'Reference', 'Model Inference', 'Bond Type Index']) # columns should reflect order in test_on_idxs of items
    return df

def bt_to_df(bt):
    # create pandas dataframe over bond type subset in given dictionary
    df = df_test_on_idxs(valid_indices_d[bt])
    df.insert(1, 'Bond Type', [bt] * len(df))
    return df

# import time
# df = pd.concat([bt_to_df(bt) for bt in valid_indices_d.keys()])
# start = time.perf_counter()
df = df_test_on_idxs(range(len(dset))) # whole dataset
# elapsed = time.perf_counter() - start
# print("elapsed time: {}".format(elapsed))
df.to_csv(dump_path, index=False)