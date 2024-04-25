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

import argparse
import random
import csv
import pickle
import os
import math
import logging
from torch.optim.lr_scheduler import ReduceLROnPlateau

# save indices for train, test split
def save_train_test(splits, path):
    with open(os.path.join(path, 'index_splits'), 'wb+') as f:
        pickle.dump(splits, f)

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

def run_trial(args):
    #refer to sum_injective_deep_batch for starting point
    dset = BDEDataset.load(args.path)
    valid_tester, train_set, splits = train_test_split(dset, device)
    save_train_test(splits, path)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    model = construct_model.get_std_sum_full(
                        injective_readout=True,
                        graph_inner_layer_sizes=args.graph_inner_layer_sizes, 
                        graph_hidden_size=args.graph_hidden_size, 
                        fc_readout_sizes=args.fc_readout_sizes, )
    model = model.to(device)
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    optim = Lion(model.parameters(), lr=args.learn_rate)
    losses = []
    vals = []

    lr_sched = ReduceLROnPlateau(optim, factor=args.reducelr_factor, patience=args.reducelr_patience, threshold=args.reducelr_threshold)
    trainer = Trainer(args.epochs, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=args.batch_size, shuffle=True), 
        lambda items: deep_attn_item_handle(items, device=device), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

if __name__ == "__main__":
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    torch.multiprocessing.set_forkserver_preload(["torch"])

    parser = argparse.ArgumentParser(description='Run DeepBDE trial with given hyperparameters, resume if train_state exists at path')
    # parser.add_argument('integers', metavar='N', type=int, nargs='+',
    #                     help='an integer for the accumulator')
    # parser.add_argument('--sum', dest='accumulate', action='store_const',
    #                     const=sum, default=max,
    #                     help='sum the integers (default: find the max)')
    parser.add_argument('--path', type=str)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--learn_rate', type=float)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--fc_readout_sizes', type=int, nargs="+")
    parser.add_argument('--graph_hidden_size', type=int)
    parser.add_argument('--graph_inner_layer_sizes', type=int, nargs="+")

    parser.add_argument('--reducelr_factor', type=float)
    parser.add_argument('--reducelr_patience', type=int)
    parser.add_argument('--reducelr_threshold', type=float)

    args = parser.parse_args()
    run_trial(args)
    # print(args)
    # print(vars(args))
    # print(args.learn_rate)