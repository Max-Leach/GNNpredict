"""
Hyperparameter optimisation for DeepBDE + NSPPK global features.

Usage example
-------------
python hpo_nsppk.py \
    --train_dset_path  ./nsppk_deepbde_run/dset_train \
    --val_dset_path    ./nsppk_deepbde_run/dset_val   \
    --atom_feat_size   10                              \
    --nsppk_nbits      5                               \
    --save_path        ./nsppk_hpo_run                 \
    --num_samples      30                              \
    --min_epochs       20                              \
    --epochs           '[50,100]'                      \
    --batch_size       '[16,32,64]'                    \
    --fc_initial_size  '[128,256]'                     \
    --fc_excess_width  '[64,128]'                      \
    --fc_excess_count  '[2,3,4]'                       \
    --graph_hidden_size '[32,64,128]'                  \
    --graph_inner_width '[64,128]'                     \
    --graph_inner_depth '[3,4,5]'                      \
    --graph_layer_count '[3,4,5]'                      \
    --reducelr_factor  '[0.5]'                         \
    --reducelr_patience '[10]'                         \
    --reducelr_threshold '[0.0001]'                    \
    --num_workers      0                               \
    --cpus_per_trial   2                               \
    --gpus_per_trial   0

The pre-built datasets at --train_dset_path / --val_dset_path must already
contain 35-dim global features (i.e. built by train_nsppk_deepbde.py with
the matching nsppk_nbits).  Build them once, then point every HPO run here.

Results are saved to  <save_path>/hpo_store/  and are fully resumable:
re-run the exact same command after an interruption to continue.
"""

import argparse
import json
import pickle
import os
import tempfile
import logging
from functools import partial
from pathlib import Path

import torch
from torch.nn import MSELoss
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

import ray
from ray import train, tune
import ray.cloudpickle as clpickle
from ray.tune import Checkpoint

from architecture import construct_model
from architecture.data.dataset import BDEDataset, BDESubset
from architecture.data.dataloader import RxnDataLoader
from architecture.item_handle import deep_bde_item_handle
from train.trainer import Trainer
from train.test.test_on_set import TestonSet


# ---------------------------------------------------------------------------
# Ray Tune checkpoint helpers  (identical pattern to hyper_search.py)
# ---------------------------------------------------------------------------

def _valid_reporter(valid_scores, losses, epochs_current, model, optim, lr_sched,
                    temp_dir):
    checkpoint_data = {
        "epochs_current": epochs_current,
        "losses":         losses,
        "valid_scores":   valid_scores,
        "lr_sched":       lr_sched.state_dict(),
    }
    with tempfile.TemporaryDirectory(dir=temp_dir) as checkpoint_dir:
        # model and optim are large — save separately
        for key, obj in [("model", model.state_dict()), ("optim", optim.state_dict())]:
            with open(Path(checkpoint_dir) / f"{key}.pkl", "wb") as fp:
                torch.save(obj, fp)

        with open(Path(checkpoint_dir) / "data.pkl", "wb") as fp:
            clpickle.dump(checkpoint_data, fp)

        tune.report(valid_scores[-1], checkpoint=Checkpoint.from_directory(checkpoint_dir))


def _restore_trainer_from_checkpoint(trainer):
    """Load Ray Tune checkpoint into trainer if one exists (for trial resume)."""
    checkpoint = tune.get_checkpoint()
    if not checkpoint:
        return
    with checkpoint.as_directory() as checkpoint_dir:
        with open(Path(checkpoint_dir) / "data.pkl", "rb") as fp:
            state = clpickle.load(fp)
        for key in ("model", "optim"):
            with open(Path(checkpoint_dir) / f"{key}.pkl", "rb") as fp:
                state[key] = torch.load(fp)
    trainer.restore_from_items(state)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def _train_instance(config, *, train_dset_path, val_dset_path,
                    atom_feat_size, global_feat_size, device, temp_dir, num_workers):
    """One Ray Tune trial."""

    # ── Data (fetched from Ray object store — loaded once, shared across trials) ──
    train_dset = BDEDataset.load(train_dset_path)
    val_dset   = BDEDataset.load(val_dset_path)
    train_set  = BDESubset(train_dset, list(range(len(train_dset))))
    valid_set  = BDESubset(val_dset,   list(range(len(val_dset))))

    # ── Model ─────────────────────────────────────────────────────────────────
    activfn_map = {
        'relu':      nn.ReLU,
        'elu':       nn.ELU,
        'silu':      nn.SiLU,
        'leakyrelu': nn.LeakyReLU,
        'tanh':      nn.Tanh,
    }
    activation_fn = activfn_map[config['activation_fn']]

    model = construct_model.get_std_sum_full(
        graph_inner_layer_sizes = [[config['graph_inner_width']] * config['graph_inner_depth']]
                                   * config['graph_layer_count'],
        graph_hidden_size       = config['graph_hidden_size'],
        fc_readout_sizes        = [config['fc_initial_size']]
                                   + [config['fc_excess_width']] * config['fc_excess_count'],
        activation_fn           = activation_fn,
        in_feat_sizes           = {'atom': atom_feat_size, 'bond': 7,
                                   'global': global_feat_size},
    ).to(device)

    # ── Loss / metrics ────────────────────────────────────────────────────────
    loss_fn    = MSELoss()
    metric_fns = {
        'mae':  mean_absolute_error,
        'mape': mean_absolute_percentage_error,
        'loss': lambda p, t: loss_fn(p, t).detach().item(),
    }

    handle_mod_out = lambda x: (x.to(device) * train_dset.val_stdev) + train_dset.val_mean

    valid_tester = TestonSet(
        RxnDataLoader(valid_set, batch_size=100, num_workers=num_workers),
        metric_fns,
        handle_items   = lambda items: deep_bde_item_handle(items, device=device),
        handle_mod_out = handle_mod_out,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    optim_construct    = lambda params: Adam(params, lr=config['learn_rate'])
    lr_sched_construct = lambda o: ReduceLROnPlateau(
        o,
        factor    = config['reducelr_factor'],
        patience  = config['reducelr_patience'],
        threshold = config['reducelr_threshold'],
    )

    trainer = Trainer(
        epochs             = config['epochs'],
        optim_construct    = optim_construct,
        loss_fn            = lambda p, t: loss_fn(
                                 (p.flatten() * train_set.val_stdev) + train_set.val_mean, t),
        validator          = valid_tester,
        train_loader       = RxnDataLoader(train_set, batch_size=config['batch_size'],
                                           shuffle=True, num_workers=num_workers),
        load_handle        = lambda items: deep_bde_item_handle(items, device=device),
        valid_reporter     = partial(_valid_reporter, temp_dir=temp_dir),
        lr_sched_construct = lr_sched_construct,
        model              = model,
    )

    _restore_trainer_from_checkpoint(trainer)
    trainer()


# ---------------------------------------------------------------------------
# Top-level search launcher
# ---------------------------------------------------------------------------

def run_hpo(args):
    # ── Search space ──────────────────────────────────────────────────────────
    list_params = [
        'graph_inner_width', 'graph_inner_depth', 'graph_layer_count',
        'graph_hidden_size', 'fc_initial_size', 'fc_excess_width',
        'fc_excess_count', 'activation_fn', 'reducelr_factor',
        'reducelr_patience', 'reducelr_threshold', 'epochs', 'batch_size',
    ]
    arg_dict    = vars(args)
    param_space = {p: tune.choice(arg_dict[p]) for p in list_params}
    param_space['learn_rate'] = tune.loguniform(1e-4, 1e-1)

    # ── Device ────────────────────────────────────────────────────────────────
    device = (torch.device('cuda')
              if getattr(args, 'gpus_per_trial', 0) and args.gpus_per_trial > 0
              else torch.device('cpu'))

    global_feat_size = 3 + (2 ** args.nsppk_nbits)

    trainable = tune.with_resources(
        tune.with_parameters(
            _train_instance,
            train_dset_path  = args.train_dset_path,
            val_dset_path    = args.val_dset_path,
            atom_feat_size   = args.atom_feat_size,
            global_feat_size = global_feat_size,
            device           = device,
            temp_dir         = getattr(args, 'temp_dir', None),
            num_workers      = args.num_workers,
        ),
        resources={
            "cpu": getattr(args, 'cpus_per_trial', 1),
            "gpu": getattr(args, 'gpus_per_trial', 0),
        },
    )

    store_path = Path(args.save_path) / 'hpo_store'
    if (store_path / 'tuner.pkl').exists():
        print(f"Resuming HPO from {store_path}")
        tuner = tune.Tuner.restore(os.path.abspath(store_path), trainable, param_space=param_space)
    else:
        print(f"Starting fresh HPO run → {store_path}")
        tuner = tune.Tuner(
            trainable,
            tune_config = tune.TuneConfig(
                metric      = "loss",
                mode        = "min",
                num_samples = args.num_samples,
            ),
            run_config  = tune.RunConfig(
                storage_path = os.path.abspath(args.save_path),
                name         = 'hpo_store',
            ),
            param_space = param_space,
        )

    results = tuner.fit()

    # ── Print and save best result ────────────────────────────────────────────
    best = results.get_best_result(metric="loss", mode="min")
    print("\n=== Best trial ===")
    print(f"  Config : {best.config}")
    print(f"  Metrics: {best.metrics}")

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, 'results.pkl'), 'wb') as f:
        clpickle.dump(results, f)
    with open(os.path.join(args.save_path, 'best_config.pkl'), 'wb') as f:
        pickle.dump(best.config, f)
    print(f"\nResults saved to {args.save_path}/results.pkl")
    print(f"Best config saved to {args.save_path}/best_config.pkl")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    torch.multiprocessing.set_forkserver_preload(['torch'])

    p = argparse.ArgumentParser(
        description='HPO for DeepBDE + NSPPK.  Pre-build the datasets with '
                    'train_nsppk_deepbde.py first, then point this script at them.')

    # ── Required paths ────────────────────────────────────────────────────────
    p.add_argument('--train_dset_path', type=str, required=True,
                   help='Path to saved train BDEDataset (dset_train dir)')
    p.add_argument('--val_dset_path',   type=str, required=True,
                   help='Path to saved val BDEDataset (dset_val dir)')
    p.add_argument('--save_path',       type=str, required=True,
                   help='Directory to write HPO results and checkpoints')

    # ── Feature sizes ─────────────────────────────────────────────────────────
    p.add_argument('--atom_feat_size', type=int, required=True,
                   help='Atom feature size (saved as atom_feat_size.pkl by '
                        'train_nsppk_deepbde.py)')
    p.add_argument('--nsppk_nbits',   type=int, default=5,
                   help='nbits used when building the dataset (default: 5 → 32 NSPPK dims)')

    # ── Search-space lists (pass as JSON, e.g. \'[32,64,128]\') ───────────────
    p.add_argument('--epochs',              type=json.loads, required=True)
    p.add_argument('--batch_size',          type=json.loads, required=True)
    p.add_argument('--fc_initial_size',     type=json.loads, required=True)
    p.add_argument('--fc_excess_width',     type=json.loads, required=True)
    p.add_argument('--fc_excess_count',     type=json.loads, required=True)
    p.add_argument('--graph_hidden_size',   type=json.loads, required=True)
    p.add_argument('--graph_inner_width',   type=json.loads, required=True)
    p.add_argument('--graph_inner_depth',   type=json.loads, required=True)
    p.add_argument('--graph_layer_count',   type=json.loads, required=True)
    p.add_argument('--reducelr_factor',     type=json.loads, required=True)
    p.add_argument('--reducelr_patience',   type=json.loads, required=True)
    p.add_argument('--reducelr_threshold',  type=json.loads, required=True)
    p.add_argument('--activation_fn',       type=json.loads,
                   default='["silu"]',
                   help='List of activation functions to try '
                        '(default: ["silu"])')

    # ── Ray / runtime ─────────────────────────────────────────────────────────
    p.add_argument('--num_samples',    type=int, required=True,
                   help='Total number of trials to run')
    p.add_argument('--min_epochs',     type=int, default=20,
                   help='Minimum epochs before early stopping is considered')
    p.add_argument('--num_workers',    type=int, default=0,
                   help='DataLoader workers per trial')
    p.add_argument('--cpus_per_trial', type=int, default=1)
    p.add_argument('--gpus_per_trial', type=int, default=0)
    p.add_argument('--temp_dir',       type=str, default=None,
                   help='Temp directory for Ray checkpoints (default: system temp)')

    args = p.parse_args()
    args.train_dset_path = os.path.abspath(args.train_dset_path)
    args.val_dset_path   = os.path.abspath(args.val_dset_path)
    # atom_feat_size can also be read from the saved pkl if not supplied manually
    atom_feat_pkl = os.path.join(
        os.path.dirname(args.train_dset_path), 'atom_feat_size.pkl')
    if os.path.exists(atom_feat_pkl):
        with open(atom_feat_pkl, 'rb') as f:
            saved_size = pickle.load(f)
        if saved_size != args.atom_feat_size:
            print(f"[warning] --atom_feat_size {args.atom_feat_size} differs from "
                  f"saved value {saved_size}. Using saved value.")
            args.atom_feat_size = saved_size

    run_hpo(args)
