from novel_arch.deep_attn.data.dset_generate import from_csv
from novel_arch.deep_attn.data.indices_split import compile_indices, reverse_bt_to_indices, split_by_bts
from itertools import chain
import os
import pickle

def test():
    root = '/home/moistry/Documents/research/data/dry_run/'
    dpath = '/home/moistry/Documents/research/data/ALFABET_data/acp_updated_NoDupes.csv'
    max_lines = 2000

    dset = from_csv(dpath, max_lines=max_lines, start_line=1)
    dset_path = os.path.join(root, 'fake_dset')
    dset.save(dset_path, as_lazy=True)

    bt2indices = compile_indices(dpath, max_lines=max_lines)
    rev_bt = reverse_bt_to_indices(bt2indices)
    train, test, valid = split_by_bts(bt2indices, [0.8, 0.1, 0.1])
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