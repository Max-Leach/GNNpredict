from novel_arch.deep_attn.data.dset_generate import from_csv
from novel_arch.deep_attn.data.dataloader import RxnDataLoader
from novel_arch.deep_attn.data.dataset import BDEDataset, BDESubset, train_test_split

from train.test.test_on_set import TestonSet
from train.test.eval_metrics import deep_attn_item_handle
from novel_arch.deep_attn import construct_model
from novel_arch.deep_attn.model import DeepAttn
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

def test_hpo_run():
    global human_centipede
    human_centipede = True
    class Args:
        pass
    args = Args()
    args.cpus_per_trial = 4
    args.gpus_per_trial = 0
    args.dset_path = '/home/moistry/Documents/research/data/1000_lazy/dset'
    args.train_indices_path = '/home/moistry/Documents/research/data/tune_trial/train_indices'
    args.valid_indices_path = '/home/moistry/Documents/research/data/tune_trial/valid_indices'
    args.save_path = '/home/moistry/Documents/research/data/tune_trial/'
    args.epochs = 1000
    args.num_samples = 1

    tweaker(args)

class TrainArgs:
    def __init__(self, dset_path, train_indices, valid_indices, device):
        self.dset_path = dset_path
        self.train_indices = train_indices
        self.valid_indices = valid_indices
        self.device = device

def valid_reporter(valid_scores, losses, epochs_current, model, optim, lr_sched):
    checkpoint_data = {
        "epochs_current": epochs_current,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        'lr_sched': lr_sched.state_dict(),
        'losses': losses,
        'valid_scores': valid_scores
    }
    with tempfile.TemporaryDirectory() as checkpoint_dir:
        data_path = Path(checkpoint_dir) / "data.pkl"
        with open(data_path, "wb") as fp:
            clpickle.dump(checkpoint_data, fp)

        checkpoint = Checkpoint.from_directory(checkpoint_dir)
        train.report(
            valid_scores[-1],
            checkpoint=checkpoint,
        )

def tweaker(args):
    param_config = {
        'graph_inner_width': 69,
        'graph_inner_depth': 3,
        'graph_layer_count': 3,
        'graph_hidden_size': 16,
        'fc_initial_size': 169,
        'fc_excess_size': 16, 
        'fc_excess_layers': 5,

        'learn_rate': 0.001,
        'reducelr_factor': 0.5, 
        'reducelr_patience': 15,
        'reducelr_threshold': 0.01,
        'epochs': args.epochs,
        'batch_size': 50,

    }
    with open(args.train_indices_path, 'rb') as train_indices_f:
        train_indices = pickle.load(train_indices_f)
        # print(train_indices)
    with open(args.valid_indices_path, 'rb') as valid_indices_f:
        valid_indices = pickle.load(valid_indices_f)

    if not hasattr(args, 'device') or args.device == None:
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    train_args = TrainArgs(args.dset_path, train_indices, valid_indices, device)

    scheduler = ASHAScheduler(
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2)
    
    trainable = tune.with_resources(
            tune.with_parameters(train_instance, train_args=train_args),
            resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial}
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
                log_to_file=human_centipede,
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
    
    # valid_tester, train_set, splits = train_test_split(dset, device)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}
    test_batch_size = 100 # should not affect any result, just time required to test
    num_workers = 4
    if train_args.device == None:
        handle_mod_out=lambda x: (x * main_dset.val_stdev) + main_dset.val_mean
    else:
        handle_mod_out=lambda x: (x.to(train_args.device) * main_dset.val_stdev) + main_dset.val_mean
    valid_tester = TestonSet(RxnDataLoader(valid_set, batch_size=test_batch_size, num_workers=num_workers), metric_fns, handle_items=lambda items: deep_attn_item_handle(items, device=train_args.device), handle_mod_out=handle_mod_out)
    # save_train_test(splits, args.path)

    model = construct_model.get_std_sum_full(
                        injective_readout=True,
                        graph_inner_layer_sizes=[[config['graph_inner_width']]*config['graph_inner_depth']]*config['graph_layer_count'], 
                        graph_hidden_size=config['graph_hidden_size'], 
                        fc_readout_sizes=[config['fc_initial_size']]+[config['fc_excess_layers']]*config['fc_excess_layers'], )
    model = model.to(train_args.device)
    # begin_test = valid_tester(model)
    loss_fn = MSELoss()
    # optim = Lion(model.parameters(), lr=args.learn_rate)
    losses = []
    vals = []

    optim_construct = lambda params: Adam(params, lr=config['learn_rate'])
    lr_sched_construct = lambda o: ReduceLROnPlateau(o, factor=config['reducelr_factor'], patience=config['reducelr_patience'], threshold=config['reducelr_threshold'])

    trainer = Trainer(config['epochs'], optim_construct, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=num_workers), 
        lambda items: deep_attn_item_handle(items, device=train_args.device), 
        valid_reporter=valid_reporter,
        # iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, args.path), 
        lr_sched_construct=lr_sched_construct,
        # epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']),
        # save_dir=args.path,
        # should_stop=lambda epochs_current, valid_scores: should_stop_if_no_mae_decrease(epochs_current, valid_scores, args.min_epochs, args.epochs_of_no_mae_drop_before_stop),
        model=model,
        )
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            data_path = Path(checkpoint_dir) / "data.pkl"
            with open(data_path, "rb") as fp:
                checkpoint_state = pickle.load(fp)
            trainer.restore_from_items(checkpoint_state)
    trainer()

###################### ----------- OLD TRAINABLE FUNCTION --------------- ###############
def eval_on_config(config, dset_ref, indices_ref, model_construct):
    model = model_construct(config)
    loss_fn = MSELoss()
    optim_select = {
        'adam': lambda lr: Adam(model.parameters(), lr=lr*10), # <---!!!! adam does better with higher lr
        'lion': lambda lr: Lion(model.parameters(), lr=lr),
    }
    op = optim_select[config['optim']](lr=config['lr'])
    dset = ray.get(dset_ref)
    indices = ray.get(indices_ref)
    subset = BDESubset(dset, indices)
    valid_tester, train_set, _ = train_test_split(subset, test_batch_size=400, num_workers=1)

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        op.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0
    
    lr_sched = ReduceLROnPlateau(op, factor=config['lrs_factor'], patience=config['lrs_patience'], threshold=config['lrs_threshold'])

    train_loader = RxnDataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=1)
    trainer = Trainer(epochs=config['epochs'], 
                    optim_construct=lambda params: op, 
                    loss_fn=lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), 
                    validator=valid_tester,
                    train_loader=train_loader,
                    load_handle=deep_attn_item_handle,
                    valid_reporter=valid_reporter,
                    epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']),
                    )
    trainer(model)