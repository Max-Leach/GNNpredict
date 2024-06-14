import argparse
import json
import random
import csv
import pickle
import os
import math
import logging
import torch

from novel_arch.deep_attn import hyper_search

if __name__ == "__main__":
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    torch.multiprocessing.set_forkserver_preload(["torch"])

    parser = argparse.ArgumentParser(description='Run DeepBDE trial with given hyperparameters, resume if train_state exists at path')

    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--dset_path', type=str, required=True)
    parser.add_argument('--train_indices_path', type=str, required=True)
    parser.add_argument('--valid_indices_path', type=str, required=True)
    parser.add_argument('--epochs', type=json.loads, required=True)
    parser.add_argument('--learn_rate', type=json.loads, required=True)
    parser.add_argument('--batch_size', type=json.loads, required=True)

    parser.add_argument('--fc_initial_size', type=json.loads, required=True)
    parser.add_argument('--fc_excess_width', type=json.loads, required=True)
    parser.add_argument('--fc_excess_count', type=json.loads, required=True)

    parser.add_argument('--graph_hidden_size', type=json.loads, required=True)

    parser.add_argument('--graph_inner_width', type=json.loads, required=True)
    parser.add_argument('--graph_inner_depth', type=json.loads, required=True)
    parser.add_argument('--graph_layer_count', type=json.loads, required=True)

    parser.add_argument('--reducelr_factor', type=json.loads, required=True)
    parser.add_argument('--reducelr_patience', type=json.loads, required=True)
    parser.add_argument('--reducelr_threshold', type=json.loads, required=True)
    parser.add_argument('--cpus_per_trial', type=int, required=False)
    parser.add_argument('--gpus_per_trial', type=int, required=False)
    parser.add_argument('--temp_dir', type=str, required=False)    
    
    parser.add_argument('--min_epochs', type=int, required=True)
    parser.add_argument('--num_samples', type=int, required=True)
    # parser.add_argument('--epochs_of_no_mae_drop_before_stop', type=int, required=True)

    args = parser.parse_args()
    # if train_state exists, check arg signature to see if matches current one
    # otherwise it is a bad restart
    args_d = vars(args).copy()
    arg_ignore_list = ['device', 'path', 'dset_path', 'temp_dir', 'train_indices_path', 'valid_indices_path']
    for a in arg_ignore_list:
        args_d.pop(a, None) # this is so device can be changed with same hyperparameters
    if os.path.exists(os.path.join(args.save_path, 'arg_signature')):
        with open(os.path.join(args.save_path, 'arg_signature'), 'rb') as arg_f:
            arg_sig = pickle.load(arg_f)
            assert arg_sig == args_d, 'Pre-existing save but arguments do not match!'
    else:
        with open(os.path.join(args.save_path, 'arg_signature'), 'wb+') as arg_f:
            pickle.dump(args_d, arg_f)
    hyper_search.tweaker(args)