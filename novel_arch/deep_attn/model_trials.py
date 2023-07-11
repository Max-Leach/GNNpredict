from novel_arch.deep_attn.data.dset_generate import from_csv
from novel_arch.deep_attn.data.dataloader import RxnDataLoader
from novel_arch.deep_attn.data.dataset import BDEDataset, BDESubset

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
import logging
from novel_arch.deep_attn import hp_op
from torch.optim.lr_scheduler import ReduceLROnPlateau

def attn_sum():
    dset = BDEDataset.load('/home/preet/data/20000/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_attn_model(sum_like=True)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    optim = Lion(model.parameters(), lr=0.0002)

    lr_sched = ReduceLROnPlateau(optim, factor=0.5, patience=30)
    trainer = Trainer(180, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=84, shuffle=True), 
        deep_attn_item_handle, 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def std_model_no_sum():
    dset = BDEDataset.load('/home/preet/data/20000/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_model()
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    optim = Lion(model.parameters(), lr=0.0002)

    lr_sched = ReduceLROnPlateau(optim, factor=0.5, patience=30)
    trainer = Trainer(180, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=84, shuffle=True), 
        deep_attn_item_handle, 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def std_model_sum():
    dset = BDEDataset.load('/home/preet/data/20000/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum()
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    optim = Lion(model.parameters(), lr=0.0002)

    lr_sched = ReduceLROnPlateau(optim, factor=0.5, patience=30)
    trainer = Trainer(180, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=84, shuffle=True), 
        deep_attn_item_handle, 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def std_model_sum_full():
    dset = BDEDataset.load('/home/preet/data/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum_full()
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    optim = Lion(model.parameters(), lr=0.0002)

    lr_sched = ReduceLROnPlateau(optim, factor=0.5, patience=30)
    trainer = Trainer(1000, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=100, shuffle=True), 
        deep_attn_item_handle, 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def std_model_sum_full_dropout():
    dset = BDEDataset.load('/home/preet/data/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum_full(
                        graph_inner_layer_sizes=[[256]*4]*5, 
                        graph_hidden_size=128, 
                        fc_readout_sizes=[256]+[128]*3, 
                        dropout=0.1)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    optim = Lion(model.parameters(), lr=0.0002)

    lr_sched = ReduceLROnPlateau(optim, factor=0.5, patience=30)
    trainer = Trainer(1000, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=128, shuffle=True), 
        deep_attn_item_handle, 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def run_model_trial(arg):
    models = {
        'attn_sum' : attn_sum,
        'std_model_no_sum' : std_model_no_sum,
        'std_model_sum' : std_model_sum,
        'std_model_sum_full' : std_model_sum_full,
        'std_model_sum_full_dropout' : std_model_sum_full_dropout,
    }
    models[arg]()