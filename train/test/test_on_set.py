from train.eval_metrics import deep_attn_item_handle, eval_metrics_over_loader
from novel_arch.deep_attn.data.dataloader import RxnDataLoader

# test a model with same conditions every time
class TestonSet:
    def __init__(self, loader, metric_fns: dict, handle_items=deep_attn_item_handle, handle_mod_out=lambda x: x):
        self.loader = loader
        self.metric_fns = metric_fns
        self.handle_items = handle_items
        self.handle_mod_out = handle_mod_out

    def __call__(self, model):
        return eval_metrics_over_loader(model, self.loader, self.metric_fns, self.handle_items, self.handle_mod_out)