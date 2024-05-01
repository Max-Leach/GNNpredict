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
import json
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

def should_stop_if_no_mae_decrease(epochs_current, valid_scores, min_epochs, num_epochs_of_no_drop):
    if epochs_current < min_epochs or epochs_current < num_epochs_of_no_drop:
        return False
    thres = valid_scores[-num_epochs_of_no_drop]['mae']
    for vs in valid_scores[-num_epochs_of_no_drop:]:
        if vs['mae'] < thres:
            return False
    return True

def run_trial(args):
    #refer to sum_injective_deep_batch for starting point
    main_dset = BDEDataset.load(args.dset_path)
    with open(args.train_indices_path, 'rb') as train_indices_f:
        train_indices = pickle.load(train_indices_f)
        # print(train_indices)
    with open(args.valid_indices_path, 'rb') as valid_indices_f:
        valid_indices = pickle.load(valid_indices_f)
    train_set = BDESubset(main_dset, train_indices)
    valid_set = BDESubset(main_dset, valid_indices)
    if not hasattr(args, 'device') or args.device == None:
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    # valid_tester, train_set, splits = train_test_split(dset, device)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}
    test_batch_size = 100 # should not affect any result, just time required to test
    num_workers = 0
    if device == None:
        handle_mod_out=lambda x: (x * main_dset.val_stdev) + main_dset.val_mean
    else:
        handle_mod_out=lambda x: (x.to(device) * main_dset.val_stdev) + main_dset.val_mean
    valid_tester = TestonSet(RxnDataLoader(valid_set, batch_size=test_batch_size, num_workers=num_workers), metric_fns, handle_items=lambda items: deep_attn_item_handle(items, device=device), handle_mod_out=handle_mod_out)
    # save_train_test(splits, args.path)

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

    optim_construct = lambda params: Lion(params, lr=args.learn_rate)
    lr_sched_construct = lambda o: ReduceLROnPlateau(optim, factor=args.reducelr_factor, patience=args.reducelr_patience, threshold=args.reducelr_threshold)
    trainer = Trainer(args.epochs, optim_construct, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=args.batch_size, shuffle=True), 
        lambda items: deep_attn_item_handle(items), 
        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, args.path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, args.path), 
        lr_sched_construct=lr_sched_construct,
        # epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']),
        save_dir=args.path,
        should_stop=lambda epochs_current, valid_scores: should_stop_if_no_mae_decrease(epochs_current, valid_scores, args.min_epochs, args.epochs_of_no_mae_drop_before_stop),
        model=model,
        )
    trainer()

if __name__ == "__main__":
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    torch.multiprocessing.set_forkserver_preload(["torch"])

    parser = argparse.ArgumentParser(description='Run DeepBDE trial with given hyperparameters, resume if train_state exists at path')

    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--dset_path', type=str, required=True)
    parser.add_argument('--train_indices_path', type=str, required=True)
    parser.add_argument('--valid_indices_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--learn_rate', type=float, required=True)
    parser.add_argument('--batch_size', type=int, required=True)

    parser.add_argument('--fc_readout_sizes', type=json.loads, required=True)
    parser.add_argument('--graph_hidden_size', type=int, required=True)
    parser.add_argument('--graph_inner_layer_sizes', type=json.loads, required=True)

    parser.add_argument('--reducelr_factor', type=float, required=True)
    parser.add_argument('--reducelr_patience', type=int, required=True)
    parser.add_argument('--reducelr_threshold', type=float, required=True)
    parser.add_argument('--device', type=str, required=False)
    
    parser.add_argument('--min_epochs', type=int, required=True)
    parser.add_argument('--epochs_of_no_mae_drop_before_stop', type=int, required=True)

    args = parser.parse_args()
    # if train_state exists, check arg signature to see if matches current one
    # otherwise it is a bad restart
    if os.path.exists(os.path.join(args.path, 'train_state')):
        with open(os.path.join(args.path, 'arg_signature'), 'rb') as arg_f:
            arg_sig = pickle.load(arg_f)
            assert arg_sig == vars(args), 'Pre-existing save but arguments do not match!'
    else:
        with open(os.path.join(args.path, 'arg_signature'), 'wb+') as arg_f:
            pickle.dump(vars(args), arg_f)
    run_trial(args)