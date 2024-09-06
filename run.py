from novel_arch.deep_attn.train import run_trial
import argparse
import json
import pickle
import os
import logging
import torch

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
    parser.add_argument('--atom_feat_size', type=int, required=False)

    parser.add_argument('--fc_readout_sizes', type=json.loads, required=True)
    parser.add_argument('--graph_hidden_size', type=int, required=True)
    parser.add_argument('--graph_inner_layer_sizes', type=json.loads, required=True)
    parser.add_argument('--activation_fn', type=str, required=False)

    parser.add_argument('--reducelr_factor', type=float, required=True)
    parser.add_argument('--reducelr_patience', type=int, required=True)
    parser.add_argument('--reducelr_threshold', type=float, required=True)
    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--num_workers', type=int, required=True)
    
    parser.add_argument('--min_epochs', type=int, required=True)
    parser.add_argument('--epochs_of_no_mae_drop_before_stop', type=int, required=True)

    args = parser.parse_args()
    # if train_state exists, check arg signature to see if matches current one
    # otherwise it is a bad restart
    args_d = vars(args).copy()
    arg_ignore_list = ['device', 'path', 'dset_path', 'train_indices_path', 'valid_indices_path', 'num_workers']
    for a in arg_ignore_list:
        args_d.pop(a, None) # this is so device can be changed with same hyperparameters
    if os.path.exists(os.path.join(args.path, 'early_stopped')):
        print('Trial has already been early stopped!')
        exit()
    if os.path.exists(os.path.join(args.path, 'train_state')):
        with open(os.path.join(args.path, 'arg_signature'), 'rb') as arg_f:
            arg_sig = pickle.load(arg_f)
            assert arg_sig == args_d, 'Pre-existing save but arguments do not match!'
    else:
        with open(os.path.join(args.path, 'arg_signature'), 'wb+') as arg_f:
            pickle.dump(args_d, arg_f)

    if not hasattr(args, 'atom_feat_size') or args.atom_feat_size == None:
        args.atom_feat_size = 18

    run_trial(args)