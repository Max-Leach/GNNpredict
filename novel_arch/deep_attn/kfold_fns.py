from train.test.k_fold import KFoldCV
from lion_pytorch import Lion
from torch.nn import MSELoss
from novel_arch.deep_attn.data.dataloader import RxnDataLoader
from novel_arch.deep_attn.data.dataset import BDESubset, BDEDataset
from train.test.eval_metrics import deep_attn_item_handle
from torch.optim.lr_scheduler import ReduceLROnPlateau
from novel_arch.deep_attn import construct_model
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from train.test.test_on_set import TestonSet
import os
import pickle

def sum_injective_batch(args):
    dump_path = args[0]

    folds = 10
    dset_path = '/home/preet/data/dset_lazy/dset'
    dset = BDEDataset.load(dset_path)
    dset_len = len(dset)
    stdev = dset.val_stdev
    mean = dset.val_mean
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}
    handle_mod_out=lambda x: (x * stdev) + mean
    create_validator = lambda tloader: TestonSet(tloader, metric_fns, handle_items=lambda items: deep_attn_item_handle(items), handle_mod_out=handle_mod_out)
    kfold = KFoldCV(folds, lambda: BDEDataset.load(dset_path),
                dset_len=dset_len, epochs=600, optim_construct=lambda p: Lion(p, lr=0.000015),
                loss_fn=lambda p,t: loss_fn((p.flatten() *stdev) +mean, t), train_loader_construct=lambda tr_set, i: RxnDataLoader(BDESubset(tr_set, i), shuffle=True, batch_size=200, num_workers=1),
                test_loader_construct=lambda tset, i: RxnDataLoader(BDESubset(tset, i), batch_size=200, num_workers=1),
                load_handle=deep_attn_item_handle,
                test_handle_mod_out=lambda x: (x *stdev) +mean,
                metric_fns=metric_fns,
                create_validator=create_validator)
    # lr_sched = ReduceLROnPlateau(optim, factor=0.8, patience=15, threshold=1e-2)
    # epoch_fn = lambda scores, epoch: lr_sched.step(scores['loss'])
    model_construct = lambda: construct_model.get_std_sum_full(injective_readout=True)
    result = kfold.run_on_model_construct(model_construct, lambda lr_sched: lambda scores, epoch: lr_sched.step(scores['loss']), lr_sched_construct=lambda op: ReduceLROnPlateau(op, factor=0.8, patience=15, threshold=1e-2), path=dump_path)
    with open(os.path.join(dump_path, 'result'), 'wb+') as f:
        pickle.dump(result, f)