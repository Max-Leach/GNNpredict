from novel_arch.deep_attn.data.dset_generate import from_csv
from novel_arch.deep_attn.data.dataloader import RxnDataLoader
from novel_arch.deep_attn.data.dataset import BDEDataset, BDESubset

from train.test.test_on_set import TestonSet
from train.test.eval_metrics import deep_attn_item_handle
from novel_arch.deep_attn import construct_model
# from novel_arch.deep_attn.model import DeepAttn
# from novel_arch.deep_attn.feat_type_updaters import concat_sum_atom_edge_feat, aggreg_atom_edge_no_repeat, AttnNodeEdgeAggreg, AtomEdgeReducer, bond_mean, atom_mean, bond_sum, atom_sum, A2GReducer, B2GReducer
from novel_arch.deep_attn.model_trials import run_model_trial

from lion_pytorch import Lion
from torch.optim import Adam
from torch.nn import MSELoss
import torch
from train.trainer import Trainer

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

import random
import csv
import logging
logging.basicConfig(format='%(message)s', level=logging.DEBUG)
from novel_arch.deep_attn import hp_op
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys

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

import pickle
def save_full_dataset():
    dset = from_csv('acp_updated_NoDupes.csv', start_line=1)
    with open('converted.pkl', 'wb') as dset_file:
        pickle.dump(dset, dset_file)

def hp_trash(name):
    paff = '/home/preet/data/8000/dset'
    dset = BDEDataset.load(paff)
    all_indices = list(range(len(dset)))
    indices = all_indices[:]
    hp_op.tweaker(name, dset, indices)

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    opers = {
        'hp_trash' : lambda arg, remain_args : hp_trash(arg),
        'trial' : lambda arg, remain_args : run_model_trial(arg, remain_args),
    }

    selector = sys.argv[1]
    try:
        arg = sys.argv[2]
    except IndexError:
        arg = None
    try:
        remain_args = sys.argv[3:]
    except IndexError:
        remain_args = []
    opers[selector](arg, remain_args)