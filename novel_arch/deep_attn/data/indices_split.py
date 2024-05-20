import csv
import random
import copy
from functools import reduce
from itertools import chain
import math

# indices are 0 from the starting point, so by default the line at 1 is considered 0 when compiling indices
def compile_indices(path, bondtype_col=6, start_line=1, max_lines=None):
    with open(path) as f:
        r = csv.reader(f, delimiter=',')
        csv_list = tuple(r)
        if max_lines == None:
            max_lines = len(csv_list)
        bt_to_indices = {}
        for i, rxn_line in enumerate(csv_list[start_line:max_lines]):
            e = rxn_line[bondtype_col]
            bt = e
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
    each_size = [0] * len(set_sizes)
    # set_indices = [{}] * len(set_sizes)
    # find portion of bond type list to split and then pop
    for bt, idxs in bt_to_indices.items():
        random.shuffle(idxs)
        idxs_len = len(idxs)
        pop_lens = [int(idxs_len * frac) for frac in set_sizes]
        covered = sum(pop_lens)
        resid = idxs_len - covered
        assert resid < len(pop_lens)
        # distribute residual until none remaining among lowest
        lowest = sorted(list(range(resid)), key=lambda i: each_size[i])
        for i in lowest:
            pop_lens[i] += 1
        # assert reduce(lambda x,y: x*y, pop_lens) > 0 # check no element is 0
        # print(bt, sum(pop_lens), idxs_len)
        assert sum(pop_lens) == idxs_len # check that split can cover all items
        idxs = idxs.copy()
        for set_to_bti, pop_len, i in zip(set_indices, pop_lens, range(len(each_size))):
            # pop_len = round(idxs_len * frac)
            set_to_bti[bt] = idxs[-pop_len:]
            idxs = idxs[:-pop_len]
            each_size[i] += pop_len
    
    return set_indices