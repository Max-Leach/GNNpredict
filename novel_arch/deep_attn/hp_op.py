from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.air import session, Checkpoint
import ray

from novel_arch.deep_attn.model import DeepAttn
from novel_arch.deep_attn.feat_type_updaters import concat_sum_atom_edge_feat, aggreg_atom_edge_no_repeat, AttnNodeEdgeAggreg, AtomEdgeReducer, bond_mean, atom_mean, bond_sum, atom_sum, A2GReducer, B2GReducer

from lion_pytorch import Lion
from torch.optim import Adam
from torch.nn import MSELoss
import torch

from train.trainer import Trainer
from train.test.eval_metrics import deep_attn_item_handle
from train.test.test_on_set import TestonSet
from novel_arch.deep_attn.data.dset_generate import from_csv
from novel_arch.deep_attn.data.dataloader import RxnDataLoader
from novel_arch.deep_attn.data.dataset import BDEDataset, BDESubset
from novel_arch.deep_attn import construct_model

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import random

import yaml
import os

# param_setting references key in yaml file
def tweaker(hyper_setting, dset, indices):
    config = get_config(hyper_setting)
    def model_on_config(config: dict):
        # return construct_model.get_attn_model(
        #     fc_readout_sizes=[128]+[64]*config['fc_excess_layers'], 
        #     graph_inner_layer_sizes=[[config['graph_inner_width']]*config['graph_inner_depth']]*config['graph_layer_count'], 
        #     graph_hidden_size=config['graph_hidden_size'],
        #     internal_attn_size=config['internal_attn_size'],
        #     sum_like=True)
        return construct_model.get_std_sum(
            fc_readout_sizes=[128]+[64]*config['fc_excess_layers'], 
            graph_inner_layer_sizes=[[config['graph_inner_width']]*config['graph_inner_depth']]*config['graph_layer_count'], 
            graph_hidden_size=config['graph_hidden_size'],
            dropout=config['dropout'])
    tweak_model_on_config(model_on_config, config, num_samples=config['num_samples'], dset=dset, indices=indices)

# retrieve hyperparam config via yaml file
def get_config(key):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, 'hyperparams.yaml')
    with open(path, 'r') as yml_f:
        yml = yaml.safe_load(yml_f)
        config = yml[key]
        parse = {
            'range' : lambda d: tune.choice(range(d[0], d[1])),
            'range2' : lambda d: tune.choice([2**i for i in range(d[0], d[1])]),
            'loguni' : lambda d: tune.loguniform(d[0], d[1]),
            'uni' : lambda d: tune.uniform(d[0], d[1]),
            '' : lambda d: tune.choice(d),
        }
        for k, dat in config.items():
            if type(dat) is tuple or type(dat) is list:
                dtype, raw = dat
                config[k] = parse[dtype](raw)
            else:
                config[k] = dat
        return config

def valid_reporter(scores, losses, epoch, model, optim):
    checkpoint_data = {
        "epoch": epoch,
        "net_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
    }
    checkpoint = Checkpoint.from_dict(checkpoint_data)

    session.report(
            scores, # assume scores will have 'mape' and 'loss' keys
            checkpoint=checkpoint)

def eval_on_config(config, dset_ref, indices_ref, model_construct):
    model = model_construct(config)
    loss_fn = MSELoss()
    # op = Adam(model.parameters(), lr=config['lr'])
    op = Lion(model.parameters(), lr=config['lr'])
    dset = ray.get(dset_ref)
    indices = ray.get(indices_ref)
    subset = BDESubset(dset, indices)
    _, valid_tester, train_set = get_sets(subset)

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    train_loader = RxnDataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    trainer = Trainer(epochs=config['epochs'], 
                    optim_construct=lambda params: op, 
                    loss_fn=lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), 
                    validator=valid_tester,
                    train_loader=train_loader,
                    load_handle=deep_attn_item_handle,
                    valid_reporter=valid_reporter
                    )
    trainer(model)
    
def get_sets(dset):
    all_indices = list(range(len(dset)))
    random.shuffle(all_indices)
    split = int(0.9 * len(all_indices))
    train_set = BDESubset(dset, all_indices[:split])
    test_set = BDESubset(dset, all_indices[split:])

    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    valid_tester = TestonSet(RxnDataLoader(test_set, batch_size=100), metric_fns, handle_mod_out=lambda x: (x * test_set.val_stdev) + test_set.val_mean)
    train_loader = RxnDataLoader(train_set, batch_size=32, shuffle=True)
    return train_loader, valid_tester, train_set

def tweak_model_on_config(model_construct, config, indices, num_samples=3, dset=None):
    if dset == None:
        dset = from_csv('/home/pmistry/Documents/research/data/ALFABET_data/acp_updated_NoDupes.csv', max_lines=800, start_line=1)
    indices_ref = ray.put(indices)
    dset_ref = ray.put(dset)

    scheduler = AsyncHyperBandScheduler(time_attr='training_iteration', max_t=1000, grace_period=16, metric='loss', mode='min', reduction_factor=2)
    result = tune.run(
                lambda config: eval_on_config(config, dset_ref, indices_ref, model_construct), 
                config=config, 
                num_samples=num_samples, 
                scheduler=scheduler
                )

    best_trial = result.get_best_trial("mae", "min", "last")
    print('Final Results')
    print('best trial', best_trial.config)
    for m_n in ['loss', 'mape', 'mae']:
        print('final {} in trial'.format(m_n), best_trial.last_result[m_n])

    with open('best_results.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow([str({n : best_trial.last_result[n] for n in ['loss', 'mae', 'mape']}), str(best_trial.config)])