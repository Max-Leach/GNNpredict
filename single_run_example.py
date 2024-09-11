from novel_arch.deep_attn.data.dset_generate import from_csv
from novel_arch.deep_attn.data.dataloader import RxnDataLoader
from novel_arch.deep_attn.data.dataset import BDEDataset, BDESubset

from train.test.test_on_set import TestonSet
from novel_arch.deep_attn.item_handle import deep_bde_item_handle
from novel_arch.deep_attn import construct_model
from novel_arch.deep_attn.model import DeepBDE
from novel_arch.deep_attn.feat_type_updaters import concat_sum_atom_edge_feat, bond_mean, atom_mean, bond_sum, atom_sum, A2GReducer, B2GReducer

from lion_pytorch import Lion
from torch.optim import Adam
from torch.nn import MSELoss
import torch
from train.trainer import Trainer

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

import random
import csv
import pickle
import os
import math
import logging
import dgl

if __name__ == '__main__':
    import torch
    torch.manual_seed(0)
    import random
    random.seed(0)
    import numpy as np
    np.random.seed(0) 
    
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    path = '/home/moistry/Documents/research/data/trainer_trial'

    dset = BDEDataset.load('/home/moistry/Documents/research/data/1000_lazy/dset')
    dset.load_graphs = True
    # valid_tester, train_set, splits = train_test_split(dset)
    # # save_train_test(splits, path)
    # loss_fn = MSELoss()
    # losses = []
    # vals = []
    # metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum_full(
                        graph_inner_layer_sizes=[[128]*5]*6, 
                        graph_hidden_size=64, 
                        fc_readout_sizes=[256]+[128]*5, )

    # run a sample
    model.eval()
    (g, f, rxn, _), t = dset[0]
    # rxn._final_graph = 0
    f = {nt: torch.tensor(f[nt], dtype=torch.float) for nt in ['bond', 'atom', 'global']}
    p = model(dgl.batch(g), f, [rxn])
    p = (p * dset.val_stdev) * dset.val_mean
    print(p)
