import logging

## train inputted model same way each time
class Trainer:
    def __init__(self, epochs, optim_construct, loss_fn, validator, train_loader, load_handle):
        self.optim_construct = optim_construct # construct optimizer on model
        self.loss_fn = loss_fn # can customize to reshape outputs properly
        self.validator = validator # tester type of class
        self.train_loader = train_loader
        self.epochs = epochs
        self.load_handle = load_handle # take raw input from data loader and put into form directly useable by model

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
                losses[-1].append(loss.detach().item())
            valid_score = self.validator(model)
            if valid_score != None:
                logging.info('>> validation after epoch {} : {}'.format(e, valid_score))
        return losses