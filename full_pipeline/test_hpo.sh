#!/bin/bash

source ../env/bin/activate

RUN_PATH='/home/moistry/Documents/research/GNNpredict/hpo.py'

python $RUN_PATH \
        --save_path '/home/moistry/Documents/research/data/tune_trial/' \
        --dset_path '/home/moistry/Documents/research/data/old_stuff/dset_lazy/dset' \
        --train_indices_path '/home/moistry/Documents/research/data/dry_run/train_indices_286' \
        --valid_indices_path '/home/moistry/Documents/research/data/dry_run/valid_indices_286' \
        --cpus_per_trial 4 \
        --gpus_per_trial 0 \
        --num_workers 3 \
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
        --epochs '[100]' \
        --batch_size '[64, 128, 256, 512]' \
        \
        --reducelr_factor '[0.8]' \
        --reducelr_patience '[15]' \
        --reducelr_threshold '[0.01]' \
        \
        --min_epochs 10 \
        --num_samples 4 \