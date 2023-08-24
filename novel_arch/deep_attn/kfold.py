from novel_arch.deep_attn import kfold_fns

def kfold_trial(arg, remain_args):
    trials = {fname : getattr(kfold_fns, fname) for fname in dir(kfold_fns)}
    trials[arg](remain_args)