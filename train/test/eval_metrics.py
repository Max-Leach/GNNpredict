import copy

# handle items from deep_attn dataloader
def deep_attn_item_handle(items, device=None):
    dat, val = items
    graphs, feats, rxn_feat_gens, _ = dat
    
    if device != None:
        graphs = graphs.to(device)
        for n in feats.keys():
            feats[n] = feats[n].to(device)
    # rxn_feat_gens = copy.deepcopy(rxn_feat_gens) # to prevent dangling references as rxn_feat_gen gets modified during a forward pass
    return (graphs, feats, rxn_feat_gens), val.to(device) # just omit the idxs entry

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
            metrics[m_name].append(m_fn(vals.flatten().detach().cpu().numpy(), preds.flatten().detach().cpu().numpy()) * item_count)
        tot_count += item_count
        del mod_in, preds, vals
    return {m_name : sum(met) / tot_count for m_name, met in metrics.items()}