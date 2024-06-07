#!/bin/bash

source ../env/bin/activate

RUN_PATH='/home/moistry/Documents/research/GNNpredict/hpo.py'

python $RUN_PATH \
        --save_path '/home/moistry/Documents/research/data/tune_trial/' \
        --dset_path '/home/moistry/Documents/research/data/dry_run/fake_dset/dset' \
        --train_indices_path '/home/moistry/Documents/research/data/dry_run/train_indices' \
        --valid_indices_path '/home/moistry/Documents/research/data/dry_run/test_indices' \
        --cpus_per_trial 4 \
        --gpus_per_trial 0 \
        \
        --graph_inner_width '[128, 256]' \
        --graph_inner_depth '[2, 3, 4]' \
        --graph_layer_count '[2, 3, 4]' \
        \
        --graph_hidden_size '[128, 64, 32]' \
        \
        --fc_initial_size '[256, 128]' \
        --fc_excess_width '[256, 128, 64]' \
        --fc_excess_count '[3, 4]' \
        \
        --learn_rate '[0.002, 0.001, 0.0005, 0.0001]' \
        --epochs '[5]' \
        --batch_size '[64, 128, 256, 512]' \
        \
        --reducelr_factor '[0.8]' \
        --reducelr_patience '[15]' \
        --reducelr_threshold '[0.01]' \
        \
        --min_epochs 1 \
        --num_samples 2 \