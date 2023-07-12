import logging

## train inputted model same way each time
class Trainer:
    def __init__(self, epochs, optim_construct, loss_fn, validator, train_loader, load_handle, iter_reporter=lambda loss, model, e, i: None, valid_reporter=lambda scores, losses, epoch, model, optim: None, epoch_fn=lambda scores, epoch: None):
        self.optim_construct = optim_construct # construct optimizer on model
        self.loss_fn = loss_fn # can customize to reshape outputs properly
        self.validator = validator # tester type of class
        self.iter_reporter = iter_reporter # run at every iteration of training
        self.valid_reporter = valid_reporter # this will run on valid scores from validator above
        self.train_loader = train_loader
        self.epochs = epochs
        self.load_handle = load_handle # take raw input from data loader and put into form directly useable by model
        self.epoch_fn = epoch_fn # runs at end of each epoch

    def __call__(self, model):
        logging.info('---- initiating train cycle ----')
        optim = self.optim_construct(model.parameters())
        losses = []
        for e in range(self.epochs):
            losses.append([])
            logging.info('---- on epoch {} ----'.format(e))
            for i, dat in enumerate(self.train_loader):
                mod_in, targ = self.load_handle(dat)
                model.train()
                pred = model(*mod_in)
                loss = self.loss_fn(pred, targ)
                logging.info('{} - loss {}'.format(i, loss.detach().item()))
                optim.zero_grad()
                loss.backward()
                optim.step()
                self.iter_reporter(loss.detach().item(), model, e, i)
                losses[-1].append(loss.detach().item())
            valid_score = self.validator(model)
            self.epoch_fn(valid_score, e)
            self.valid_reporter(valid_score, losses, e, model, optim)
            if valid_score != None:
                logging.info('>> validation after epoch {} : {}'.format(e, valid_score))
        return losses