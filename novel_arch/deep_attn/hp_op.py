from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.air import session, Checkpoint

from novel_arch.deep_attn.model import DeepAttn
from novel_arch.deep_attn.feat_type_updaters import concat_sum_atom_edge_feat, aggreg_atom_edge_no_repeat, AttnNodeEdgeAggreg, AtomEdgeReducer, bond_mean, atom_mean, bond_sum, atom_sum, A2GReducer, B2GReducer

from lion_pytorch import Lion
from torch.optim import Adam
from torch.nn import MSELoss
import torch

from train.trainer import Trainer
from train.test.eval_metrics import deep_attn_item_handle
from train.test.test_on_set import TestonSet
from novel_arch.deep_attn.data.from_csv import bdedataset_from_csv
from novel_arch.deep_attn.data.dataloader import RxnDataLoader
from novel_arch.deep_attn.data.dataset import BDEDataset, BDESubset

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import random

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

def eval_on_config(config, valid_tester, train_set):
    # very rudimentary, we need things like variable inner graph size
    model = DeepAttn(
                atom_aggregators=concat_sum_atom_edge_feat,
                b2g_aggregator=bond_mean(),
                a2g_aggregator=atom_mean(),
                in_feat_sizes={'atom': 12, 'bond': 7, 'global': 3},
                graph_hidden_size=config['graph_hidden_size'],
                graph_layers=config['graph_layer_count'],
                graph_inner_layer_sizes=[[config['graph_inner_width']] * config['graph_inner_depth']] * config['graph_layer_count'],
                residual=True,
                fc_readout_sizes=[128] + [64] * config['fc_excess_layers'],
            )
    loss_fn = MSELoss()
    # op = Adam(model.parameters(), lr=config['lr'])
    op = Lion(model.parameters(), lr=config['lr'])

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
    
def get_dataset(line_cap=800):
    dset = bdedataset_from_csv('/home/pmistry/Documents/research/data/ALFABET_data/acp_updated_NoDupes.csv', max_lines=line_cap, start_line=1)
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

def main():
    _, valid_tester, train_set = get_dataset()

    config = {
        "graph_hidden_size": tune.choice([2**i for i in range(3, 6)]),
        "graph_layer_count": tune.choice([i for i in range(2, 6)]),
        "graph_inner_width": tune.choice([2**i for i in range(4,7)]),
        "graph_inner_depth": tune.choice(tuple(range(1,5))),
        "lr": tune.loguniform(0.9e-4, 2.3e-3),
        "epochs": tune.choice(tuple(range(40, 60))),
        "batch_size": tune.choice([16, 32, 64]),
        "fc_excess_layers": tune.choice(tuple(range(1,4)))
    }
    scheduler = AsyncHyperBandScheduler(time_attr='training_iteration', max_t=100, metric='loss', mode='min', reduction_factor=2)
    result = tune.run(
                lambda config: eval_on_config(config, valid_tester, train_set), 
                config=config, 
                num_samples=45, 
                scheduler=scheduler
                )

    best_trial = result.get_best_trial("mae", "min", "last")
    print('Final Results')
    print('best trial', best_trial.config)
    for m_n in ['loss', 'mape', 'mae']:
        print('final {} in trial'.format(m_n), best_trial.last_result[m_n])

    # config = {
    #     "graph_hidden_size": 32,
    #     "graph_layer_count": 3,
    #     "lr": 0.001,
    #     # "batch_size": tune.choice([2, 4, 8, 16, 32, 64]),
    # }
    # eval_on_config(config, train_loader, valid_tester, train_set)