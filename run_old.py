import sys
import os
# from novel_arch.deep_attn import hyper_optim
from novel_arch.deep_attn.model_trials import run_model_trial
# from novel_arch.deep_attn.kfold import kfold_trial
import logging
import torch

if __name__ == '__main__':
    os.environ['TUNE_DISABLE_AUTO_CALLBACK_LOGGERS'] = '1'

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    torch.multiprocessing.set_forkserver_preload(["torch"])

    opers = {
        # 'hyper_optim' : lambda arg, remain_args : hyper_optim.run_optim(arg, remain_args),
        'trial' : lambda arg, remain_args : run_model_trial(arg, remain_args),
        # 'kfold' : lambda arg, remain_args : kfold_trial(arg, remain_args),
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