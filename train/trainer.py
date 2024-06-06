import logging
import torch
import pickle
import os

## train inputted model same way each time
# restart from previously saved if save_dir is not none and a training state is found
# assumes model, opt, and lr sched are all same type
class Trainer:
    def __init__(self, epochs, optim_construct, loss_fn, validator, train_loader, load_handle, iter_reporter=lambda loss, model, e, i: None, valid_reporter=lambda scores, losses, epoch, model, optim, lr_sched: None, epoch_fn=lambda scores, epoch: None, save_dir=None, lr_sched_construct=None, should_stop=None, model=None):
        self.optim_construct = optim_construct # construct optimizer on model
        self.loss_fn = loss_fn # can customize to reshape outputs properly
        self.validator = validator # tester type of class
        self.iter_reporter = iter_reporter # run at every iteration of training
        self.valid_reporter = valid_reporter # this will run on valid scores from validator above
        self.train_loader = train_loader
        self.total_epochs = epochs
        self.load_handle = load_handle # take raw input from data loader and put into form directly useable by model
        self.epoch_fn = epoch_fn # runs at end of each epoch
        self.save_dir = save_dir
        self.lr_sched_construct = lr_sched_construct
        self.model = model
        self.epochs_current = None
        self.optim = None
        self.lr_sched = None
        self.should_stop = should_stop

    def __call__(self):
        if self.save_dir != None and os.path.exists(os.path.join(self.save_dir, 'train_state')):
            self._restore()
            logging.info('---- restarting train cycle from epoch {} ----'.format(self.epochs_current))
        else:
            self.optim = self.optim_construct(self.model.parameters())
            self.epochs_current = 0
            self.lr_sched = self.lr_sched_construct(self.optim)
            self.losses = []
            self.valid_scores = []
            logging.info('---- initiating train cycle ----')
        return self._resume_train()

    def _resume_train(self):
        while self.epochs_current < self.total_epochs:
            self.losses.append([])
            logging.info('---- on epoch {} ----'.format(self.epochs_current))
            for i, dat in enumerate(self.train_loader):
                mod_in, targ = self.load_handle(dat)
                self.model.train()
                pred = self.model(*mod_in)
                loss = self.loss_fn(pred, targ)
                logging.info('{} - loss {}'.format(i, loss.detach().item()))
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.iter_reporter(loss.detach().item(), self.model, self.epochs_current, i)
                self.losses[-1].append(loss.detach().item())
            valid_score = self.validator(self.model)
            self.valid_scores.append(valid_score)
            self.epoch_fn(valid_score, self.epochs_current)
            self.lr_sched.step(valid_score['loss'])
            self.valid_reporter(self.valid_scores, self.losses, self.epochs_current, self.model, self.optim, self.lr_sched)
            if valid_score != None:
                logging.info('>> validation after epoch {} : {}'.format(self.epochs_current, valid_score))
            self.epochs_current += 1
            if self.save_dir != None:
                with open(os.path.join(self.save_dir, 'train_state'), 'wb') as f:
                    torch.save(
                        {
                            'optim': self.optim.state_dict(), 
                            'lr_sched': self.lr_sched.state_dict(),
                            'model': self.model.state_dict(), 
                            'epochs_current': self.epochs_current, 
                            'losses': self.losses,
                            'valid_scores': self.valid_scores
                        }, 
                        f)
            if self.should_stop != None and self.should_stop(self.epochs_current, self.valid_scores):
                with open(os.path.join(self.save_dir, 'early_stopped'), 'wb+') as f:
                    pickle.dump('epoch {}'.format(self.epochs_current), f)
                break
        return self.losses

    def restore_from_items(self, items):
        self.model.load_state_dict(items['model'])
        self.optim = self.optim_construct(self.model.parameters())
        self.optim.load_state_dict(items['optim'])
        self.lr_sched = self.lr_sched_construct(self.optim)
        self.lr_sched.load_state_dict(items['lr_sched'])
        self.epochs_current = items['epochs_current']
        self.losses = items['losses']
        self.valid_scores = items['valid_scores']

    def _restore(self):
        with open(os.path.join(self.save_dir, 'train_state'), 'rb') as f:
            items = torch.load(f)
            self.restore_from_items(items)