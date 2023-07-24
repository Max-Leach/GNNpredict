import torch
import model_trial_fns

def run_model_trial(arg, remain_args):
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        dev = torch.device('cpu')
    models = {fname : getattr(model_trial_fns, fname) for fname in dir(model_trial_fns)}
    models[arg](remain_args, device=dev)