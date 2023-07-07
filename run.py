from novel_arch.deep_attn.data.dset_generate import from_csv
from novel_arch.deep_attn.data.dataloader import RxnDataLoader
from novel_arch.deep_attn.data.dataset import BDEDataset, BDESubset

from train.test.test_on_set import TestonSet
from train.test.eval_metrics import deep_attn_item_handle
from novel_arch.deep_attn import construct_model
# from novel_arch.deep_attn.model import DeepAttn
# from novel_arch.deep_attn.feat_type_updaters import concat_sum_atom_edge_feat, aggreg_atom_edge_no_repeat, AttnNodeEdgeAggreg, AtomEdgeReducer, bond_mean, atom_mean, bond_sum, atom_sum, A2GReducer, B2GReducer

from lion_pytorch import Lion
from torch.optim import Adam
from torch.nn import MSELoss
import torch
from train.trainer import Trainer

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

import random
import logging
logging.basicConfig(format='%(message)s', level=logging.DEBUG)
from novel_arch.deep_attn import hp_op
from torch.optim.lr_scheduler import ReduceLROnPlateau

def kfold():
    dset = from_csv('/home/pmistry/Documents/research/data/ALFABET_data/acp_updated_NoDupes.csv', max_lines=64, start_line=1)
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error}
    loss_fn = MSELoss()

    from train.test.k_fold import KFoldCV
    kfold = KFoldCV(10, dset, 25, 
            optim_construct=lambda p: Adam(p, lr=0.001), 
            loss_fn=lambda p,t: loss_fn((p.flatten() * dset.val_stdev) + dset.val_mean, t), 
            train_loader_construct=lambda d, i: RxnDataLoader(BDESubset(d, i), batch_size=32, shuffle=True),
            test_loader_construct=lambda d, i: RxnDataLoader(BDESubset(d, i), batch_size=100),
            load_handle=deep_attn_item_handle,
            test_handle_mod_out=lambda x: (x * dset.val_stdev) + dset.val_mean,
            metric_fns=metric_fns
            )
    test, valids = kfold.run_on_model_construct(
        lambda: DeepAttn(
                    atom_aggregators=concat_sum_atom_edge_feat,
                    b2g_aggregator=bond_mean(),
                    a2g_aggregator=atom_mean(),
                    in_feat_sizes={'atom': 12, 'bond': 7, 'global': 3},
                    graph_hidden_size=64,
                    graph_layers=3,
                    graph_inner_layer_sizes=[[64]] * 3,
                    residual=True
                ))
    total = sum([item[1] for item in test['mape']])
    sum_test = sum([item[0] * item[1] for item in test['mape']])
    print('total score mape', sum_test / total, 'total elements k fold', total)

def train_select():
    dset = from_csv('/home/pmistry/Documents/research/data/ALFABET_data/acp_updated_NoDupes.csv', max_lines=800, start_line=1)
    # subset = BDESubset(dset, tuple(range(800)))
    # newset = BDEDataset.from_subset(subset)
    # dset = newset

    # dset = BDEDataset.load('dset')
    _, valid_tester, train_set = hp_op.get_sets(dset)
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}

    # model = construct_model.get_std_model()
    model = construct_model.get_attn_model()
    begin_test = valid_tester(model)
    loss_fn = MSELoss()
    # op = Adam(model.parameters(), lr=0.001)
    optim = Lion(model.parameters(), lr=0.0001)

    lr_sched = ReduceLROnPlateau(optim, factor=0.5, patience=30, threshold=3)
    trainer = Trainer(4, lambda p: optim, lambda p,t: loss_fn((p.flatten() * train_set.val_stdev) + train_set.val_mean, t), valid_tester, 
        RxnDataLoader(train_set, batch_size=65, shuffle=True), 
        deep_attn_item_handle, 
        epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss']))
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', valid_tester(model))

import pickle
def save_full_dataset():
    dset = from_csv('acp_updated_NoDupes.csv', start_line=1)
    with open('converted.pkl', 'wb') as dset_file:
        pickle.dump(dset, dset_file)

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    # save_full_dataset()
    hp_op.tweaker('initial')
    # train_select()