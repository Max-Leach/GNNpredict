import logging
import torch
import os

## train inputted model same way each time
class Trainer:
    def __init__(self, epochs, optim_construct, loss_fn, validator, train_loader, load_handle, iter_reporter=lambda loss, model, e, i: None, valid_reporter=lambda scores, losses, epoch, model, optim: None, epoch_fn=lambda scores, epoch: None, save_dir=None, lr_sched_construct=None):
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
        self.epochs_current = None
        self.optim = None
        self.lr_sched = None
        self.model = None

    def __call__(self, model):
        assert self.model == None, "looks like this Trainer object is already being used"
        logging.info('---- initiating train cycle ----')
        self.optim = self.optim_construct(model.parameters())
        self.epochs_current = 0
        self.model = model
        self.lr_sched = self.lr_sched_construct(self.optim)
        self.losses = []
        return self.resume_train()
        # while self.epochs_completed < self.total_epochs:
        #     losses.append([])
        #     logging.info('---- on epoch {} ----'.format(self.epochs_completed))
        #     for i, dat in enumerate(self.train_loader):
        #         mod_in, targ = self.load_handle(dat)
        #         model.train()
        #         pred = model(*mod_in)
        #         loss = self.loss_fn(pred, targ)
        #         logging.info('{} - loss {}'.format(i, loss.detach().item()))
        #         optim.zero_grad()
        #         loss.backward()
        #         optim.step()
        #         self.iter_reporter(loss.detach().item(), model, self.epochs_completed, i)
        #         losses[-1].append(loss.detach().item())
        #     valid_score = self.validator(model)
        #     self.epoch_fn(valid_score, self.epochs_completed)
        #     self.valid_reporter(valid_score, losses, self.epochs_completed, model, optim)
        #     if valid_score != None:
        #         logging.info('>> validation after epoch {} : {}'.format(self.epochs_completed, valid_score))
        #     self.epochs_completed += 1
        # return losses

    def resume_train(self):
        assert self.model != None, "nothing to resume"
        logging.info('---- training cycle from epoch {} ----'.format(self.epochs_current))
        # self.optim = self.optim_construct(model.parameters())
        # self.model = model
        # self.losses = []
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
            self.epoch_fn(valid_score, self.epochs_current)
            self.lr_sched.step(valid_score['loss'])
            self.valid_reporter(valid_score, self.losses, self.epochs_current, self.model, self.optim)
            if valid_score != None:
                logging.info('>> validation after epoch {} : {}'.format(self.epochs_current, valid_score))
            self.epochs_current += 1
            if self.save_dir != None:
                with open(os.path.join(self.save_dir, 'train_state'), 'wb') as f:
                    torch.save(
                        {
                            'optim': self.optim.state_dict(), 
                            'lr_sched': self.lr_sched.state_dict(),
                            'model': self.model, 
                            'epochs_current': self.epochs_current, 
                            'losses': self.losses
                        }, 
                        f)
        return self.losses

    def restore(self, save_dir):
        with open(os.path.join(self.save_dir, 'train_state'), 'rb') as f:
            items = torch.load(f)
            # self.optim = items['optim']
            self.model = items['model']
            self.optim = self.optim_construct(self.model.parameters())
            self.optim.load_state_dict(items['optim'])
            self.lr_sched = self.lr_sched_construct(self.optim)
            self.lr_sched.load_state_dict(items['lr_sched'])
            self.epochs_current = items['epochs_current']
            self.losses = items['losses']