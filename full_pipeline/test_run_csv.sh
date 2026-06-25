#!/bin/bash

source ../env/bin/activate

RUN_PATH='/home/moistry/Documents/research/GNNpredict/run_csv.py'

python $RUN_PATH \
        --path '/home/moistry/Documents/research/data/dry_run/model_run' \
        --dset_path '/home/moistry/Documents/research/data/dry_run/fake_dset/dset' \
        --train_indices_path '/home/moistry/Documents/research/data/dry_run/train_indices' \
        --valid_indices_path '/home/moistry/Documents/research/data/dry_run/valid_indices' \
        --device 'cpu' \
        \
        --num_workers 1 \
        --csv_path '/home/moistry/Documents/research/data/hyperparams/results analysis/progress/params_sorted_progress_aggregated_epoch-30.csv'\
        --param_row 0 \
        \
        --min_epochs 5 \
        --epochs_of_no_mae_drop_before_stop 10 \