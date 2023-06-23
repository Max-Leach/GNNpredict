# handle items from deep_attn dataloader
def deep_attn_item_handle(items):
    dat, val = items
    return dat[:3], val # just omit the idxs entry

## handle items - take raw output from loader, return value and valid model inputs
def eval_metrics_over_loader(model, loader, metric_fns: dict, handle_items=deep_attn_item_handle, handle_mod_out=lambda x: x):
    model.eval()
    metrics = {m_name : [] for m_name in metric_fns}
    tot_count = 0
    for items in iter(loader):
        mod_in, vals = handle_items(items)
        preds = handle_mod_out(model(*mod_in).detach())
        item_count = len(vals)
        for m_name, m_fn in metric_fns.items():
            metrics[m_name].append(m_fn(vals.flatten(), preds.flatten()) * item_count)
        tot_count += item_count
    return {m_name : sum(met) / tot_count for m_name, met in metrics.items()}