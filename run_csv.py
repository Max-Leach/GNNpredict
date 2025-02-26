from architecture.train import run_trial
import argparse
import json
import pickle
import os
import logging
import torch
import pandas as pd

if __name__ == "__main__":
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    torch.multiprocessing.set_forkserver_preload(["torch"])

    parser = argparse.ArgumentParser(description='Run DeepBDE trial with given hyperparameters, resume if train_state exists at path')

    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--dset_path', type=str, required=True)
    parser.add_argument('--train_indices_path', type=str, required=True)
    parser.add_argument('--valid_indices_path', type=str, required=True)

    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--param_row', type=int, required=True)

    parser.add_argument('--device', type=str, required=False)
    parser.add_argument('--num_workers', type=int, required=True)
    parser.add_argument('--min_epochs', type=int, required=True)
    parser.add_argument('--epochs_of_no_mae_drop_before_stop', type=int, required=True)

    args = parser.parse_args()
    # if train_state exists, check arg signature to see if matches current one
    # otherwise it is a bad restart
    args_d = vars(args).copy()
    arg_ignore_list = ['device', 'path', 'dset_path', 'train_indices_path', 'valid_indices_path', 'num_workers', 'csv_path']
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
    
    # add parameters from the csv file
    df = pd.read_csv(args.csv_path)
    df = df.drop('Name', axis=1).drop('Loss', axis=1, errors='ignore').drop('Comments', axis=1, errors='ignore').drop('All Comments', axis=1, errors='ignore')
    params = df.iloc[args.param_row]
    def conv(n):
        strs = ['activation_fn']
        floats = ['learn_rate', 'reducelr_factor','reducelr_patience','reducelr_threshold']
        if n in strs:
            return str
        elif n in floats:
            return float
        else:
            return int
    params = {n : conv(n)(v) for n, v in params.items()}
    print(params)
    args.__dict__ = {**vars(args), **params}
    args.graph_inner_layer_sizes = [[args.graph_inner_width]*args.graph_inner_depth]*args.graph_layer_count
    args.fc_readout_sizes = [args.fc_initial_size]+[args.fc_excess_width]*args.fc_excess_count
    args.atom_feat_size = 18 # make sure change this when testing/not-testing on smaller dataset!

    run_trial(args)