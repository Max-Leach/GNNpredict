from novel_arch.deep_attn.data.dset_generate import from_csv
from novel_arch.deep_attn.data.dataloader import RxnDataLoader
from novel_arch.deep_attn.data.dataset import BDEDataset, BDESubset, train_test_split

from train.test.test_on_set import TestonSet
from train.test.eval_metrics import deep_attn_item_handle
from novel_arch.deep_attn import construct_model
from novel_arch.deep_attn.model import DeepBDE
from novel_arch.deep_attn.feat_type_updaters import concat_sum_atom_edge_feat, aggreg_atom_edge_no_repeat, AttnNodeEdgeAggreg, AtomEdgeReducer, bond_mean, atom_mean, bond_sum, atom_sum, A2GReducer, B2GReducer

from lion_pytorch import Lion
from torch.optim import Adam
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from train.trainer import Trainer

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

import ray.cloudpickle as clpickle
import pickle
import tempfile
from pathlib import Path
import os
from functools import partial

class TrainArgs:
    def __init__(self, dset_path, train_indices, valid_indices, device, temp_dir, num_workers):
        self.dset_path = dset_path
        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.device = device
        self.temp_dir = temp_dir
        self.num_workers = num_workers

def valid_reporter(valid_scores, losses, epochs_current, model, optim, lr_sched, temp_dir):
    checkpoint_data = {
        "epochs_current": epochs_current,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        'lr_sched': lr_sched.state_dict(),
        'losses': losses,
        'valid_scores': valid_scores
    }
    with tempfile.TemporaryDirectory(dir=temp_dir) as checkpoint_dir:
        chonky = ['model', 'optim'] # save separately so not to approach limit on individual file sizes
        for chonk in chonky:
            chonk_path = Path(checkpoint_dir) / '{}.pkl'.format(chonk)
            with open(chonk_path, "wb") as fp:
                # clpickle.dump(checkpoint_data[chonk], fp)
                torch.save(checkpoint_data[chonk], fp)
            del checkpoint_data[chonk]
        
        data_path = Path(checkpoint_dir) / 'data.pkl'
        with open(data_path, "wb") as fp:
            clpickle.dump(checkpoint_data, fp)

        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        train.report(
            valid_scores[-1],
            checkpoint=checkpoint,
        )

def tweaker(args):
    hyperparams = ['graph_inner_width',
        'graph_inner_depth',
        'graph_layer_count',
        'graph_hidden_size',
        'fc_initial_size',
        'fc_excess_width', 
        'fc_excess_count',
        'learn_rate',
        'reducelr_factor', 
        'reducelr_patience',
        'reducelr_threshold',
        'epochs',
        'batch_size']
    arg_dict = vars(args)
    param_config = { h : tune.choice(arg_dict[h]) for h in hyperparams }

    with open(args.train_indices_path, 'rb') as train_indices_f:
        train_indices = pickle.load(train_indices_f)
        # print(train_indices)
    with open(args.valid_indices_path, 'rb') as valid_indices_f:
        valid_indices = pickle.load(valid_indices_f)

    if not hasattr(args, 'gpus_per_trial') and args.gpus_per_trial > 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if hasattr(args, 'temp_dir'):
        temp_dir = args.temp_dir
    else:
        temp_dir = None

    train_args = TrainArgs(args.dset_path, train_indices, valid_indices, device, temp_dir, args.num_workers)

    scheduler = ASHAScheduler(
        max_t=max(args.epochs),
        grace_period=args.min_epochs,
        reduction_factor=2)
    
    trainable = tune.with_resources(
            tune.with_parameters(train_instance, train_args=train_args),
            resources={
                        "cpu": args.cpus_per_trial, 
                    #    "gpu": args.gpus_per_trial
                    }
        )
    store_name = 'hpo_store'
    store_path = Path(args.save_path) / store_name
    if (store_path / 'tuner.pkl').exists():
        tuner = tune.Tuner.restore(
            str(store_path),
            trainable,
            param_space=param_config,
        )
    else:
        tuner = tune.Tuner(
            trainable,
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                scheduler=scheduler,
                num_samples=args.num_samples,
            ),
            run_config=train.RunConfig(
                storage_path=args.save_path,
                name=store_name,
                log_to_file=False,
            ),
            param_space=param_config,
        )

    results = tuner.fit()
    with open(os.path.join(args.save_path, 'results'), 'wb+') as f:
        clpickle.dump(results, f)

def train_instance(config, train_args):
    main_dset = BDEDataset.load(train_args.dset_path)
    train_set = BDESubset(main_dset, train_args.train_indices)
    valid_set = BDESubset(main_dset, train_args.valid_indices)
    
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}
    test_batch_size = 100 # should not affect any result, just time required to test
    # num_workers = 1
    if train_args.device == None:
        handle_mod_out=lambda x: (x * main_dset.val_stdev) + main_dset.val_mean
    else:
        handle_mod_out=lambda x: (x.to(train_args.device) * main_dset.val_stdev) + main_dset.val_mean
    valid_tester = TestonSet(RxnDataLoader(valid_set, batch_size=test_batch_size, num_workers=train_args.num_workers), metric_fns, handle_items=lambda items: deep_attn_item_handle(items, device=train_args.device), handle_mod_out=handle_mod_out)

    model = construct_model.get_std_sum_full(
                        injective_readout=True,
                        graph_inner_layer_sizes=[[config['graph_inner_width']]*config['graph_inner_depth']]*config['graph_layer_count'], 
                        graph_hidden_size=config['graph_hidden_size'], 
                        fc_readout_sizes=[config['fc_initial_size']]+[config['fc_excess_width']]*config['fc_excess_count'], )
    model = model.to(train_args.device)
    loss_fn = MSELoss()
    losses = []
    vals = []

    optim_construct = lambda params: Adam(params, lr=config['learn_rate'])
    lr_sched_construct = lambda o: ReduceLROnPlateau(o, factor=config['reducelr_factor'], patience=config['reducelr_patience'], threshold=config['reducelr_threshold'])

    trainer = Trainer(config['epochs'], optim_construct, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=train_args.num_workers), 
        lambda items: deep_attn_item_handle(items, device=train_args.device), 
        valid_reporter=partial(valid_reporter, temp_dir=train_args.temp_dir),
        # iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, args.path), 
        lr_sched_construct=lr_sched_construct,
        # epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']),
        # save_dir=args.path,
        # should_stop=lambda epochs_current, valid_scores: should_stop_if_no_mae_decrease(epochs_current, valid_scores, args.min_epochs, args.epochs_of_no_mae_drop_before_stop),
        model=model,
        )
    
    restore_trainer_from_checkpoint(trainer)

    trainer()

# keep variables out of scope by using separate function to ensure deletion
def restore_trainer_from_checkpoint(trainer):
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)

            chonky = ['model', 'optim'] # save separately so not to approach limit on individual file sizes

            for chonk in chonky:
                chonk_path = Path(checkpoint_dir) / '{}.pkl'.format(chonk)
                with open(chonk_path, "rb") as fp:
                    # chonk_state = clpickle.load(fp)
                    chonk_state = torch.load(fp)
                checkpoint_state[chonk] = chonk_state
                
            trainer.restore_from_items(checkpoint_state)