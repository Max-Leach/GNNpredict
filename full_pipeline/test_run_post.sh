#!/bin/bash

source ../env/bin/activate

RUN_PATH='/home/moistry/Documents/research/GNNpredict/run_csv.py'

python $RUN_PATH \
        --path '/home/moistry/Documents/research/data/dry_run/model_run' \
        --dset_path '/home/moistry/Documents/research/data/dry_run/fake_dset/dset_new/dataset/dset' \
        --train_indices_path '/home/moistry/Documents/research/data/dry_run/fake_dset/dset_new/train_indices' \
        --valid_indices_path '/home/moistry/Documents/research/data/dry_run/fake_dset/dset_new/valid_indices' \
        --device 'cpu' \
        \
        --num_workers 1 \
        --csv_path '/home/moistry/Documents/research/data/02-11 cleaned dset/post_tune-02_25--1.csv'\
        --param_row 0 \
        \
        --min_epochs 5 \
        --epochs_of_no_mae_drop_before_stop 10 \