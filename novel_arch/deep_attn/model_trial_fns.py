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
import pickle
import os
import math
import logging
from novel_arch.deep_attn import hp_op
from torch.optim.lr_scheduler import ReduceLROnPlateau

# standard reporter for every training iteration
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

def attn_sum(args, device):
    dset = BDEDataset.load('/home/preet/data/20000/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset, device)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_attn_model(sum_like=True)
    model = model.to(device)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    optim = Lion(model.parameters(), lr=0.0002)
    losses = []
    vals = []

    lr_sched = ReduceLROnPlateau(optim, factor=0.5, patience=30)
    trainer = Trainer(180, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=84, shuffle=True), 
        lambda items: deep_attn_item_handle(items, device=device), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def std_model_no_sum(args, device):
    path = args[0]

    dset = BDEDataset.load('/home/preet/data/20000/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset, device)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_model()
    model = model.to(device)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    optim = Lion(model.parameters(), lr=0.0002)
    losses = []
    vals = []

    lr_sched = ReduceLROnPlateau(optim, factor=0.5, patience=30)
    trainer = Trainer(180, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=84, shuffle=True), 
        lambda items: deep_attn_item_handle(items, device=device), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def std_model_sum(args, device):
    path = args[0]

    dset = BDEDataset.load('/home/preet/data/20000/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset, device)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum()
    model = model.to(device)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    optim = Lion(model.parameters(), lr=0.0002)
    losses = []
    vals = []

    lr_sched = ReduceLROnPlateau(optim, factor=0.5, patience=30)
    trainer = Trainer(180, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=84, shuffle=True), 
        lambda items: deep_attn_item_handle(items, device=device), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def std_model_sum_full(args, device):
    path = args[0]

    dset = BDEDataset.load('/home/preet/data/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset, device)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum_full()
    model = model.to(device)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    # optim = Lion(model.parameters(), lr=0.00001)
    optim = Lion(model.parameters(), lr=0.000008)
    losses = []
    vals = []

    lr_sched = ReduceLROnPlateau(optim, factor=0.5, patience=30)
    trainer = Trainer(1000, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=100, shuffle=True), 
        lambda items: deep_attn_item_handle(items, device=device), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def std_model_sum_full_attn(args, device):
    path = args[0]

    dset = BDEDataset.load('/home/preet/data/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset, device)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum_full_attn()
    model = model.to(device)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    optim = Lion(model.parameters(), lr=0.00001)
    losses = []
    vals = []

    lr_sched = ReduceLROnPlateau(optim, factor=0.5, patience=30)
    trainer = Trainer(1000, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=100, shuffle=True), 
        lambda items: deep_attn_item_handle(items, device=device), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def sum_full_deeper(args, device):
    path = args[0]

    dset = BDEDataset.load('/home/preet/data/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset, device)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum_full_deeper()
    model = model.to(device)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    # optim = Lion(model.parameters(), lr=0.00001)
    optim = Adam(model.parameters(), lr=0.0001)
    losses = []
    vals = []

    lr_sched = ReduceLROnPlateau(optim, factor=0.8, patience=8, threshold=1e-2)
    trainer = Trainer(1000, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=100, shuffle=True), 
        lambda items: deep_attn_item_handle(items, device=device), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def sum_full_dropout(args, device):
    path = args[0]

    dset = BDEDataset.load('/home/preet/data/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset, device)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum_full_deeper(dropout=0.1,
                                        graph_inner_layer_sizes=[[256]*7]*7, 
                                        graph_hidden_size=128, 
                                        fc_readout_sizes=[512]+[256]*8, )
    model = model.to(device)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    # optim = Lion(model.parameters(), lr=0.000002)
    optim = Adam(model.parameters(), lr=0.00005)
    losses = []
    vals = []

    lr_sched = ReduceLROnPlateau(optim, factor=0.5, patience=30)
    trainer = Trainer(1000, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=100, shuffle=True), 
        lambda items: deep_attn_item_handle(items, device=device), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def sum_full_injective(args, device):
    path = args[0]

    dset = BDEDataset.load('/home/preet/data/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset, device)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum_full(injective_readout=True)
    model = model.to(device)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    optim = Lion(model.parameters(), lr=0.000015)
    losses = []
    vals = []

    lr_sched = ReduceLROnPlateau(optim, factor=0.8, patience=15, threshold=1e-2)
    trainer = Trainer(1000, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=100, shuffle=True), 
        lambda items: deep_attn_item_handle(items, device=device), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def sum_full_injective_batch(args, device):
    path = args[0]

    dset = BDEDataset.load('/home/preet/data/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset, device)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum_full(injective_readout=True)
    model = model.to(device)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    optim = Lion(model.parameters(), lr=0.000015)
    losses = []
    vals = []

    lr_sched = ReduceLROnPlateau(optim, factor=0.8, patience=15, threshold=1e-2)
    trainer = Trainer(1000, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=200, shuffle=True), 
        lambda items: deep_attn_item_handle(items, device=device), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def sum_full_injective_adam(args, device):
    path = args[0]

    dset = BDEDataset.load('/home/preet/data/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset, device)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum_full(injective_readout=True)
    model = model.to(device)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    # optim = Lion(model.parameters(), lr=0.000015)
    optim = Adam(model.parameters(), lr=0.001)
    losses = []
    vals = []

    lr_sched = ReduceLROnPlateau(optim, factor=0.8, patience=15, threshold=1e-2)
    trainer = Trainer(1000, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=100, shuffle=True), 
        lambda items: deep_attn_item_handle(items, device=device), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def sum_full_injective_deep_batch(args, device):
    path = args[0]

    dset = BDEDataset.load('/home/preet/data/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset, device)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum_full(
                        injective_readout=True,
                        graph_inner_layer_sizes=[[128]*5]*6, 
                        graph_hidden_size=64, 
                        fc_readout_sizes=[256]+[128]*5, )
    model = model.to(device)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    optim = Lion(model.parameters(), lr=0.000005)
    losses = []
    vals = []

    lr_sched = ReduceLROnPlateau(optim, factor=0.8, patience=15, threshold=1e-2)
    trainer = Trainer(1000, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=200, shuffle=True), 
        lambda items: deep_attn_item_handle(items, device=device), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def sum_full_injective_deep_adam(args, device):
    path = args[0]

    dset = BDEDataset.load('/home/preet/data/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset, device)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum_full(
                        injective_readout=True,
                        graph_inner_layer_sizes=[[128]*5]*6, 
                        graph_hidden_size=64, 
                        fc_readout_sizes=[256]+[128]*5, )
    model = model.to(device)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    # optim = Lion(model.parameters(), lr=0.000015)
    optim = Adam(model.parameters(), lr=0.0005)
    losses = []
    vals = []

    lr_sched = ReduceLROnPlateau(optim, factor=0.8, patience=15, threshold=1e-2)
    trainer = Trainer(1000, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=100, shuffle=True), 
        lambda items: deep_attn_item_handle(items, device=device), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

def sum_full_injective_deep_adam_batch(args, device):
    path = args[0]

    dset = BDEDataset.load('/home/preet/data/dset')
    _, valid_tester, train_set = hp_op.get_sets(dset, device)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum_full(
                        injective_readout=True,
                        graph_inner_layer_sizes=[[128]*5]*6, 
                        graph_hidden_size=64, 
                        fc_readout_sizes=[256]+[128]*5, )
    model = model.to(device)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    # optim = Lion(model.parameters(), lr=0.000015)
    optim = Adam(model.parameters(), lr=0.0005)
    losses = []
    vals = []

    lr_sched = ReduceLROnPlateau(optim, factor=0.8, patience=15, threshold=1e-2)
    trainer = Trainer(1000, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=200, shuffle=True), 
        lambda items: deep_attn_item_handle(items, device=device), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))