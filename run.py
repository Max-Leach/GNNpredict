import sys
from novel_arch.deep_attn import hyper_optim
from novel_arch.deep_attn.model_trials import run_model_trial
import logging

if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    # torch.multiprocessing.set_sharing_strategy('file_system')

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