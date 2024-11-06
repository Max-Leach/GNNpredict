from novel_arch.deep_attn.data.dset_generate import from_csv
from novel_arch.deep_attn.data.indices_split import compile_indices, reverse_bt_to_indices, split_by_bts
from datetime import datetime
from itertools import chain
import os
import pickle

# save dir - path for saving dset
# csv_path - csv source location
# split - list of fractions to split indices, must sum to 1
def do(save_dir, csv_path, split):
    root = save_dir
    dpath = csv_path

    dset = from_csv(dpath, start_line=1)
    dset_path = os.path.join(root, 'dataset')
    dset.save(dset_path, as_lazy=True)

    do_indices(root, dpath, split)

def do_indices(root, dpath, split):
    bt2indices = compile_indices(dpath)
    rev_bt = reverse_bt_to_indices(bt2indices)
    train, test, valid = split_by_bts(bt2indices, split)
    with open(os.path.join(root, 'train_indices_d'), 'wb+') as f:
        pickle.dump(train, f)
    with open(os.path.join(root, 'valid_indices_d'), 'wb+') as f:
        pickle.dump(valid, f)
    with open(os.path.join(root, 'test_indices_d'), 'wb+') as f:
        pickle.dump(test, f)

    with open(os.path.join(root, 'train_indices'), 'wb+') as f:
        pickle.dump(tuple(chain.from_iterable(train.values())), f)
    with open(os.path.join(root, 'valid_indices'), 'wb+') as f:
        pickle.dump(tuple(chain.from_iterable(valid.values())), f)
    with open(os.path.join(root, 'test_indices'), 'wb+') as f:
        pickle.dump(tuple(chain.from_iterable(test.values())), f)
