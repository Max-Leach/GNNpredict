import random
import math
import copy
import itertools
from train.trainer import Trainer
from train.test.test_on_set import TestonSet
import logging

# run a k fold session with a dataset on inputted model constructor
# running again will use the same folds
class KFoldCV:
    def __init__(self, num_folds, dataset, epochs, optim_construct, loss_fn, train_loader_construct, test_loader_construct, load_handle, test_handle_mod_out, metric_fns, validator=lambda m: None):
        self.num_folds = num_folds
        self.dataset = dataset
        self.epochs = epochs
        self._setup_folds()
        self.optim_construct = optim_construct
        self.loss_fn = loss_fn
        # 2 below take in dataset, indices to use
        self.train_loader_construct = train_loader_construct
        self.test_loader_construct = test_loader_construct
        self.validator = validator # if stats wanted during training, but likely not used
        self.load_handle = load_handle # for trainer to handle outputs of loader given
        self.test_handle_mod_out = test_handle_mod_out # for test construction, preprocess model output before running metrics
        self.metric_fns = metric_fns # metrics for tester, as a dict
    
    def _setup_folds(self):
        # sizing to prevent imbalances from being too great as only discrete data samples
        ideal_fold_size = len(self.dataset) / float(self.num_folds)
        overhang = len(self.dataset) % self.num_folds
        ceiled = math.ceil(ideal_fold_size)
        floored = math.floor(ideal_fold_size)
        fold_sizes = [ceiled] * overhang + [floored] * (self.num_folds - overhang)
        slicing = [0] + list(itertools.accumulate(fold_sizes))
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        self.fold_indices = [indices[slicing[i]:slicing[i+1]] for i in range(len(slicing) - 1)]

    def run_on_model_construct(self, model_construct):
        valid_scores = []
        test_scores = {mn : [] for mn in self.metric_fns}
        logging.info('------- running {} fold CV -------'.format(self.num_folds))
        for test_fold_i in range(self.num_folds):
            logging.info('>>> on fold {}'.format(test_fold_i))
            model = model_construct()
            indices = copy.deepcopy(self.fold_indices)
            test_loader = self.test_loader_construct(self.dataset, indices[test_fold_i])
            del indices[test_fold_i]
            train_loader = self.train_loader_construct(self.dataset, [idx for one_fold_idxs in indices for idx in one_fold_idxs])
            trainer = Trainer(self.epochs, self.optim_construct, self.loss_fn, self.validator, train_loader, self.load_handle)
            score = trainer(model)
            valid_scores.append(score)
            tester = TestonSet(test_loader, self.metric_fns, handle_mod_out=self.test_handle_mod_out)
            metrics = tester(model)
            logging.info('>>> fold metrics: {}'.format(metrics))
            for mn in self.metric_fns:
                test_scores[mn].append((metrics[mn], len(self.fold_indices[test_fold_i])))
        return test_scores, valid_scores