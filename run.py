import sys
from novel_arch.deep_attn import hyper_optim
from novel_arch.deep_attn.model_trials import run_model_trial
import logging
import torch

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

if __name__ == '__main__':
    dset = BDEDataset.load('/home/pmistry/Documents/research/data/1000_lazy/dset')
    dset_len = len(dset)
    stdev = dset.val_stdev
    mean = dset.val_mean
    loss_fn = MSELoss()
    metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error, 'loss': lambda p, t: loss_fn(p, t).detach().item()}
    handle_mod_out=lambda x: (x * stdev) + mean
    create_validator = lambda tloader: TestonSet(tloader, metric_fns, handle_items=lambda items: deep_attn_item_handle(items), handle_mod_out=handle_mod_out)
    kfold = KFoldCV(2, lambda: BDEDataset.load('/home/pmistry/Documents/research/data/1000_lazy/dset'),
                dset_len=dset_len, epochs=20, optim_construct=lambda p: Lion(p, lr=0.000015),
                loss_fn=lambda p,t: loss_fn((p.flatten() *stdev) +mean, t), train_loader_construct=lambda tr_set, i: RxnDataLoader(BDESubset(tr_set, i), shuffle=True, batch_size=200, num_workers=1),
                test_loader_construct=lambda tset, i: RxnDataLoader(BDESubset(tset, i), batch_size=200, num_workers=1),
                load_handle=deep_attn_item_handle,
                test_handle_mod_out=lambda x: (x *stdev) +mean,
                metric_fns=metric_fns,
                create_validator=create_validator)
    # lr_sched = ReduceLROnPlateau(optim, factor=0.8, patience=15, threshold=1e-2)
    # epoch_fn = lambda scores, epoch: lr_sched.step(scores['loss'])
    model_construct = lambda: construct_model.get_std_sum_full(injective_readout=True)
    kfold.run_on_model_construct(model_construct, lambda lr_sched: lambda scores, epoch: lr_sched.step(scores['loss']), lr_sched_construct=lambda op: ReduceLROnPlateau(op, factor=0.8, patience=15, threshold=1e-2), path='/home/pmistry/Documents/research/data/usuck')
    exit()

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    # torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_forkserver_preload(["torch"])

    opers = {
        'hyper_optim' : lambda arg, remain_args : hyper_optim.run_optim(arg, remain_args),
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