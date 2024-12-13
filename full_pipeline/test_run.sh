#!/bin/bash

source ../env/bin/activate # setup python environment

RUN_PATH='/home/moistry/Documents/research/GNNpredict/run.py'

python $RUN_PATH \
        --path '/home/moistry/Documents/research/data/dry_run/model_run' \
        --dset_path '/home/moistry/Documents/research/data/dry_run/fake_dset/dset' \
        --train_indices_path '/home/moistry/Documents/research/data/dry_run/train_indices' \
        --valid_indices_path '/home/moistry/Documents/research/data/dry_run/valid_indices' \
        --device 'cpu' \
        --num_workers 1 \
        --activation_fn 'silu' \
        --atom_feat_size 12 \
        \
        --graph_inner_layer_sizes '[[128, 128, 128, 128, 128], [128, 128, 128, 128, 128], [128, 128, 128, 128, 128], [128, 128, 128, 128, 128], [128, 128, 128, 128, 128]]' \
        --graph_hidden_size 64 \
        --fc_readout_sizes '[256, 128, 128, 128, 128, 128]' \
        \
        --learn_rate 0.001 \
        --epochs 1000 \
        --batch_size 200 \
        \
        --reducelr_factor 0.8 \
        --reducelr_patience 15 \
        --reducelr_threshold 0.01 \
        \
        --min_epochs 5 \
        --epochs_of_no_mae_drop_before_stop 10 \