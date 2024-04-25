import csv
import random
import copy
from functools import reduce
import math

# indices are 0 from the starting point, so by default the line at 1 is considered 0 when compiling indices
def compile_indices(path, bondtype_col=6, start_line=1):
    with open(path) as f:
        r = csv.reader(f, delimiter=',')
        csv_list = tuple(r)
        bt_to_indices = {}
        for i, rxn_line in enumerate(csv_list[start_line:]):
            e = rxn_line[bondtype_col]
            bt = tuple(sorted(e.replace('-','').upper()))
            if bt not in bt_to_indices:
                bt_to_indices[bt] = []
            bt_to_indices[bt].append(i)
        return bt_to_indices

# map from index to bond type
def reverse_bt_to_indices(bt_to_indices):
    indices_to_bt = {}
    for bt, idxs in bt_to_indices.items():
        for idx in idxs:
            indices_to_bt[idx] = bt
    return indices_to_bt

# bt_to_indices - bond types and indices under
# set_sizes - list of fractions to exist in final split
def split_by_bts(bt_to_indices, set_sizes):
    assert sum(set_sizes) == 1
    bt_to_indices = copy.deepcopy(bt_to_indices)

    set_indices = [{} for _ in set_sizes]
    # set_indices = [{}] * len(set_sizes)
    # find portion of bond type list to split and then pop
    for bt, idxs in bt_to_indices.items():
        random.shuffle(idxs)
        idxs_len = len(idxs)
        pop_lens = [int(idxs_len * frac) for frac in set_sizes]
        covered = sum(pop_lens)
        resid = idxs_len - covered
        assert resid < len(pop_lens)
        # distribute residual until none remaining
        for i in range(resid):
            pop_lens[i] += 1
        # assert reduce(lambda x,y: x*y, pop_lens) > 0 # check no element is 0
        # print(bt, sum(pop_lens), idxs_len)
        assert sum(pop_lens) == idxs_len # check that split can cover all items
        idxs = idxs.copy()
        for set_to_bti, pop_len in zip(set_indices, pop_lens):
            # pop_len = round(idxs_len * frac)
            set_to_bti[bt] = idxs[-pop_len:]
            idxs = idxs[:-pop_len]
    
    return set_indices

    # set_lens = [set_sizes[0]]
    # for s in set_sizes[1:]:
    #     set_lens.append(s + set_lens[-1])
    # set_indices = [{}] * len(set_sizes)
    # # sums of ratios are above for splitting indices
    # for bt in bt_to_indices.keys():
    #     for i in range(len(set_indices)):
    #         set_indices[i][bt] = 

    #     all_indices = list(range(len(dset)))
    # random.shuffle(all_indices)
    # split = int(0.9 * len(all_indices))
    # train_split = all_indices[:split]
    # test_split = all_indices[split:]
    # train_set = BDESubset(dset, train_split)
    # test_set = BDESubset(dset, test_split)