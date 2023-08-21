from ray import tune
from ray.tune.search.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
from ray.air import session, Checkpoint
import ray

from lion_pytorch import Lion
from torch.optim import Adam
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

from train.trainer import Trainer
from train.test.eval_metrics import deep_attn_item_handle
from novel_arch.deep_attn.data.dataloader import RxnDataLoader
from novel_arch.deep_attn.data.dataset import BDEDataset, BDESubset, train_test_split
from novel_arch.deep_attn import construct_model

import yaml
import os
import pickle

# param_setting references key in yaml file
def tweaker(hyper_setting, dset, indices, save_path):
    config = get_config(hyper_setting)
    def model_on_config(config: dict):
        return construct_model.get_std_sum(
            fc_readout_sizes=[256]+[128]*config['fc_excess_layers'], 
            graph_inner_layer_sizes=[[config['graph_inner_width']]*config['graph_inner_depth']]*config['graph_layer_count'], 
            graph_hidden_size=config['graph_hidden_size'],
            # dropout=config['dropout']
        )
    tweak_model_on_config(model_on_config, config, save_path=save_path, num_samples=config['num_samples'], dset=dset, indices=indices)

# retrieve hyperparam config via yaml file
def get_config(key):
    dirname = os.path.dirname(__file__)
    path = os.path.join(dirname, 'hypers', 'main.yaml')
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
    optim_select = {
        'adam': lambda lr: Adam(model.parameters(), lr=lr*10), # <---!!!! adam does better with higher lr
        'lion': lambda lr: Lion(model.parameters(), lr=lr),
    }
    op = optim_select[config['optim']](lr=config['lr'])
    dset = ray.get(dset_ref)
    indices = ray.get(indices_ref)
    subset = BDESubset(dset, indices)
    valid_tester, train_set, _ = train_test_split(subset, test_batch_size=400, num_workers=5)

    checkpoint = session.get_checkpoint()

    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["net_state_dict"])
        op.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0
    
    lr_sched = ReduceLROnPlateau(op, factor=config['lrs_factor'], patience=config['lrs_patience'], threshold=config['lrs_threshold'])

    train_loader = RxnDataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=5)
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

def tweak_model_on_config(model_construct, config, save_path, indices, num_samples=3, dset=None):
    indices_ref = ray.put(indices)
    dset_ref = ray.put(dset)

    # scheduler = AsyncHyperBandScheduler(time_attr='training_iteration', max_t=1000, grace_period=8, metric='loss', mode='min', reduction_factor=2)
    # result = tune.run(
    #             lambda config: eval_on_config(config, dset_ref, indices_ref, model_construct), 
    #             config=config, 
    #             num_samples=num_samples, 
    #             scheduler=scheduler
    #             )
    alg = TuneBOHB(metric='loss', mode='min', max_concurrent=56)
    sched = HyperBandForBOHB(time_attr='training_iteration', metric='loss', mode='min', max_t=1000)
    result = tune.run(
                lambda config: eval_on_config(config, dset_ref, indices_ref, model_construct), 
                config=config, 
                num_samples=num_samples, 
                scheduler=sched,
                search_alg=alg,
                )
    with open(os.path.join(save_path, 'results'), 'wb+') as f:
        pickle.dump(result, f)

def run_optim(arg, remain_args):
    dset = BDEDataset.load('/home/preet/data/dset_lazy/dset')
    all_indices = list(range(len(dset)))
    indices = all_indices[:]
    save_path = remain_args[0]
    tweaker(arg, dset, indices, save_path)