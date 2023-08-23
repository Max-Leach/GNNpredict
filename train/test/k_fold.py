import random
import math
import copy
import itertools
from train.trainer import Trainer
from train.test.test_on_set import TestonSet
import logging
import os
import pickle
from dask.distributed import LocalCluster, Client
import dask
import torch

# standard reporter for every training iteration
def iter_reporter(loss, model, epoch, iter, loss_list, path):
    loss_list.append(loss)
    with open(os.path.join(path, 'iter_losses'), 'wb+') as f:
        pickle.dump(loss_list, f)

def valid_reporter(valid_score, losses, e, model, val_list, path):
    with open(os.path.join(path, 'last_model'), 'wb+') as m:
        torch.save(model, m)
    val_list.append(valid_score)
    with open(os.path.join(path, 'epoch_vals'), 'wb+') as f:
        pickle.dump(val_list, f)
    try:
        with open(os.path.join(path, 'best_model_vals'), 'rb') as f:
            _epoch, vals = pickle.load(f)
    except (FileNotFoundError, EOFError):
        _epoch, vals = 0, {'mae': math.inf}
    if vals['mae'] > valid_score['mae']: # lower is better
        with open(os.path.join(path, 'best_model'), 'wb+') as m:
            torch.save(model, m)
        with open(os.path.join(path, 'best_model_vals'), 'wb+') as f:
            pickle.dump((e, valid_score), f)

# run a k fold session with a dataset on inputted model constructor
# running again will use the same folds
class KFoldCV:
    def __init__(self, num_folds, dataset_construct, dset_len, epochs, optim_construct, loss_fn, train_loader_construct, test_loader_construct, load_handle, test_handle_mod_out, metric_fns, create_validator=lambda tloader: lambda m: None):
        self.num_folds = num_folds
        self.dataset_construct = dataset_construct
        self.dset_len = dset_len
        self.epochs = epochs
        self._setup_folds()
        self.optim_construct = optim_construct
        self.loss_fn = loss_fn
        # 2 below take in dataset, indices to use
        self.train_loader_construct = train_loader_construct
        self.test_loader_construct = test_loader_construct
        self.create_validator = create_validator # if stats wanted during training, but likely not used
        self.load_handle = load_handle # for trainer to handle outputs of loader given
        self.test_handle_mod_out = test_handle_mod_out # for test construction, preprocess model output before running metrics
        self.metric_fns = metric_fns # metrics for tester, as a dict
    
    def _setup_folds(self):
        # sizing to prevent imbalances from being too great as only discrete data samples
        ideal_fold_size = self.dset_len / float(self.num_folds)
        overhang = self.dset_len % self.num_folds
        ceiled = math.ceil(ideal_fold_size)
        floored = math.floor(ideal_fold_size)
        fold_sizes = [ceiled] * overhang + [floored] * (self.num_folds - overhang)
        slicing = [0] + list(itertools.accumulate(fold_sizes))
        indices = list(range(self.dset_len))
        random.shuffle(indices)
        self.fold_indices = [indices[slicing[i]:slicing[i+1]] for i in range(len(slicing) - 1)]

    # dont forget about lr scheduler in epoch fn!
    def run_on_model_construct(self, model_construct, create_epoch_fn, lr_sched_construct, path):
        # valid_scores = []
        # test_scores = {mn : [] for mn in self.metric_fns}
        with dask.config.set({"distributed.worker.daemon": False}):
            cluster = LocalCluster(processes=True)
            client = Client(cluster)
            indices_path = os.path.join(path, 'fold_indices')
            with open(indices_path, 'wb+') as i_f:
                pickle.dump(self.fold_indices, i_f)
            logging.info('------- running {} fold CV -------'.format(self.num_folds))
            test_trains = []
            for test_fold_i in range(self.num_folds):
                fold_path = os.path.join(path, 'fold_{}'.format(test_fold_i))
                if not os.path.exists(fold_path):
                    os.makedirs(fold_path)
                def fold_i_task():
                    losses = []
                    vals = []
                    logging.info('>>> starting fold {}'.format(test_fold_i))
                    model = model_construct()
                    indices = copy.deepcopy(self.fold_indices)
                    dataset = self.dataset_construct()
                    test_loader = self.test_loader_construct(dataset, indices[test_fold_i])
                    validator = self.create_validator(test_loader)
                    del indices[test_fold_i]
                    optim = self.optim_construct(model.parameters())
                    lr_sched = lr_sched_construct(optim)
                    train_loader = self.train_loader_construct(dataset, [idx for one_fold_idxs in indices for idx in one_fold_idxs])
                    trainer = Trainer(self.epochs, lambda p: optim, self.loss_fn, validator, train_loader, self.load_handle,
                        valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, fold_path),
                        iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, fold_path), 
                        epoch_fn=create_epoch_fn(lr_sched))
                    score = trainer(model)
                    tester = TestonSet(test_loader, self.metric_fns, handle_mod_out=self.test_handle_mod_out)
                    metrics = tester(model)
                    logging.info('>>> fold metrics: {}'.format(metrics))
                    return (metrics[mn], len(self.fold_indices[test_fold_i])), score, losses, vals
                test_trains = client.submit(fold_i_task)
            return client.gather(test_trains)

# valid_reporter=lambda valid_score, losses, e, model, optim: valid_reporter(valid_score, losses, e, model, vals, path),
#         iter_reporter=lambda loss, model, e, i: iter_reporter(loss, model, e, i, losses, path), 
#         epoch_fn=lambda scores, epoch: lr_sched.step(scores['loss'])