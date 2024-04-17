from novel_arch.deep_attn.data.dset_generate import from_csv
from novel_arch.deep_attn.data.dataloader import RxnDataLoader
from novel_arch.deep_attn.data.dataset import BDEDataset, BDESubset, train_test_split

from train.test.test_on_set import TestonSet
from train.test.eval_metrics import deep_attn_item_handle
from novel_arch.deep_attn import construct_model
from novel_arch.deep_attn.model import DeepAttn
from novel_arch.deep_attn.feat_type_updaters import concat_sum_atom_edge_feat, aggreg_atom_edge_no_repeat, AttnNodeEdgeAggreg, AtomEdgeReducer, bond_mean, atom_mean, bond_sum, atom_sum, A2GReducer, B2GReducer

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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dgl

def iter_reporter(loss, model, epoch, iter, loss_list, path):
    loss_list.append(loss)
    with open(os.path.join(path, 'iter_losses'), 'wb+') as f:
        pickle.dump(loss_list, f)

def valid_reporter(valid_score, losses, e, model, val_list, path):
    with open(os.path.join(path, 'last_model'), 'wb+') as m:
        torch.save(model, m)
    val_list.append(valid_score)
    with open(os.path.join(path, 'epoch_vals'), 'wb+') as f:
        pickle.dump(val_list, f)
    try:
        with open(os.path.join(path, 'best_model_vals'), 'rb') as f:
            _epoch, vals = pickle.load(f)
    except (FileNotFoundError, EOFError):
        _epoch, vals = 0, {'mae': math.inf}
    if vals['mae'] > valid_score['mae']: # lower is better
        with open(os.path.join(path, 'best_model'), 'wb+') as m:
            torch.save(model, m)
        with open(os.path.join(path, 'best_model_vals'), 'wb+') as f:
            pickle.dump((e, valid_score), f)

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
    valid_tester, train_set, splits = train_test_split(dset)
    # save_train_test(splits, path)
    loss_fn = MSELoss()
    losses = []
    vals = []
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum_full(
                        injective_readout=True,
                        graph_inner_layer_sizes=[[128]*5]*6, 
                        graph_hidden_size=64, 
                        fc_readout_sizes=[256]+[128]*5, )

    # optim = Lion(model.parameters(), lr=0.1)
    optim_construct = lambda params: Lion(params, lr=0.001)
    lr_sched_construct = lambda o: ReduceLROnPlateau(o, factor=0.8, patience=15, threshold=1e-2)
    trainer = Trainer(10, optim_construct, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=200, shuffle=False), 
        lambda items: deep_attn_item_handle(items), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
        lr_sched_construct=lr_sched_construct,
        # epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']),
        save_dir='/home/moistry/Documents/research/data/trainer_trial/ballcock',
        model=model,
        )
    trainer()

    # run a sample
    # model.eval()
    # (g, f, rxn, _), t = dset[0]
    # rxn._final_graph = 0
    # f = {nt: torch.tensor(f[nt], dtype=torch.float) for nt in ['bond', 'atom', 'global']}
    # p = model(dgl.batch(g), f, [rxn])
    # print(p)
