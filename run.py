from novel_arch.deep_attn.data.from_csv import bdedataset_from_csv
from novel_arch.deep_attn.data.dataloader import RxnDataLoader
from novel_arch.deep_attn.data.dataset import BDEDataset, BDESubset

from train.test.test_on_set import TestonSet
from train.test.eval_metrics import deep_attn_item_handle
from novel_arch.deep_attn.model import DeepAttn
from novel_arch.deep_attn.feat_type_updaters import concat_sum_atom_edge_feat, aggreg_atom_edge_no_repeat, AttnNodeEdgeAggreg, AtomEdgeReducer, bond_mean, atom_mean, bond_sum, atom_sum, A2GReducer, B2GReducer

from torch.optim import Adam
from torch.nn import MSELoss
import torch
from train.trainer import Trainer

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

import logging
logging.basicConfig(format='%(message)s', level=logging.DEBUG)

from novel_arch.deep_attn import hp_op

if __name__ == '__main__':
    hp_op.main()
exit()

def kfold():
    dset = bdedataset_from_csv('/home/pmistry/Documents/research/data/ALFABET_data/acp_updated_NoDupes.csv', max_lines=64, start_line=1)
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
def std_train():
    model = DeepAttn(
            atom_aggregators=concat_sum_atom_edge_feat,
            b2g_aggregator=bond_mean(),
            a2g_aggregator=atom_mean(),
            in_feat_sizes={'atom': 12, 'bond': 7, 'global': 3},
            graph_hidden_size=64,
            graph_layers=3,
            graph_inner_layer_sizes=[[64]] * 3,
            residual=True
        )
    loader = RxnDataLoader(dset, batch_size=100)
    test_dset = TestonSet(loader, metric_fns, handle_mod_out=lambda x: (x * dset.val_stdev) + dset.val_mean)
    test_dataset = BDESubset(dset, [32])
    valid_dset = TestonSet(RxnDataLoader(test_dataset, batch_size=100), metric_fns, handle_mod_out=lambda x: (x * test_dataset.val_stdev) + test_dataset.val_mean)
    begin_test = test_dset(model)
    loss_fn = MSELoss()
    op = Adam(model.parameters(), lr=0.001)

    trainer = Trainer(20, lambda p: Adam(p, lr=0.001), lambda p,t: loss_fn((p.flatten() * dset.val_stdev) + dset.val_mean, t), valid_dset, RxnDataLoader(dset, batch_size=16, shuffle=True), deep_attn_item_handle)
    trainer(model)

    print('begin test result', begin_test)
    print('end test result', test_dset(model))