from novel_arch.deep_attn.data.dataloader import RxnDataLoader
from novel_arch.deep_attn.data.dataset import BDEDataset, BDESubset

from train.test.test_on_set import TestonSet
from novel_arch.deep_attn.item_handle import deep_bde_item_handle
from novel_arch.deep_attn import construct_model

from torch.optim import Adam
from torch.nn import MSELoss
from torch import nn
import torch
from train.trainer import Trainer

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, max_error, root_mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pickle
import os
import math

# save indices for train, test split
def save_train_test(splits, path):
    with open(os.path.join(path, 'index_splits'), 'wb+') as f:
        pickle.dump(splits, f)

# standard reporter for every training iteration
def iter_reporter(loss, model, epoch, iter, loss_list, path):
    loss_list.append(loss)
    with open(os.path.join(path, 'iter_losses'), 'wb+') as f:
        pickle.dump(loss_list, f)

def valid_reporter(valid_scores, losses, e, model, val_list, path):
    valid_score = valid_scores[-1]
    with open(os.path.join(path, 'last_model'), 'wb+') as m:
        torch.save(model, m)
    val_list.append(valid_score)
    with open(os.path.join(path, 'epoch_vals'), 'wb+') as f:
        pickle.dump(val_list, f)
    try:
        with open(os.path.join(path, 'best_model_vals'), 'rb') as f:
            _epoch, vals = pickle.load(f)
    except (FileNotFoundError, EOFError):
        _epoch, vals = 0, {'loss': math.inf}
    if (e % 25) == 0: # save model every so epochs
        with open(os.path.join(path, 'model_epoch-{}'.format(e)), 'wb+') as m:
            torch.save(model, m)
    if vals['loss'] > valid_score['loss']: # lower is better
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

def wrap_to_numpy(func):
    return lambda t1, t2: func(t1.numpy(), t2.numpy())

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
    
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item(),
        'max_error': wrap_to_numpy(max_error), 'rmse': wrap_to_numpy(root_mean_squared_error), 'r2_score': wrap_to_numpy(r2_score), 'stdev_error': lambda p, t: torch.std(torch.abs(p - t)).detach().item(),
    }
    test_batch_size = 100 # should not affect any result, just time required to test
    # num_workers = 4
    if device == None:
        handle_mod_out=lambda x: (x * main_dset.val_stdev) + main_dset.val_mean
    else:
        handle_mod_out=lambda x: (x.to(device) * main_dset.val_stdev) + main_dset.val_mean
    valid_tester = TestonSet(RxnDataLoader(valid_set, batch_size=test_batch_size, num_workers=args.num_workers), metric_fns, handle_items=lambda items: deep_bde_item_handle(items, device=device), handle_mod_out=handle_mod_out)

    if hasattr(args, 'activation_fn') and args.activation_fn != None:
        activfn_repo = {
            'relu' : nn.ReLU,
            'elu' : nn.ELU,
            'silu' : nn.SiLU,
            'leakyrelu' : nn.LeakyReLU,
            'tanh' : nn.Tanh,
        }
        activation_fn = activfn_repo[args.activation_fn]
    else:
        activation_fn = None

    model = construct_model.get_std_sum_full(
                        graph_inner_layer_sizes=args.graph_inner_layer_sizes, 
                        graph_hidden_size=args.graph_hidden_size, 
                        fc_readout_sizes=args.fc_readout_sizes, 
                        activation_fn=activation_fn,
                        in_feat_sizes={'atom': args.atom_feat_size, 'bond': 7, 'global': 3},
                        )
    model = model.to(device)
    loss_fn = MSELoss()
    losses = []
    vals = []

    optim_construct = lambda params: Adam(params, lr=args.learn_rate)
    lr_sched_construct = lambda o: ReduceLROnPlateau(o, factor=args.reducelr_factor, patience=args.reducelr_patience, threshold=args.reducelr_threshold)
    trainer = Trainer(args.epochs, optim_construct, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers), 
        lambda items: deep_bde_item_handle(items, device=device), 
        valid_reporter=lambda valid_score, losses, e, model, optim, lr_sched: valid_reporter(valid_score, losses, e, model, vals, args.path),
        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, args.path), 
        lr_sched_construct=lr_sched_construct,
        # epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']),
        save_dir=args.path,
        should_stop=lambda epochs_current, valid_scores: should_stop_if_no_mae_decrease(epochs_current, valid_scores, args.min_epochs, args.epochs_of_no_mae_drop_before_stop),
        model=model,
        )
    trainer()