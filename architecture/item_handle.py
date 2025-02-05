# handle items from deepbde dataloader
def deep_bde_item_handle(items, device=None):
    dat, val = items
    graphs, feats, rxn_feat_gens, _ = dat
    
    if device != None:
        graphs = graphs.to(device)
        for n in feats.keys():
            feats[n] = feats[n].to(device)
    # rxn_feat_gens = copy.deepcopy(rxn_feat_gens) # to prevent dangling references as rxn_feat_gen gets modified during a forward pass
    return (graphs, feats, rxn_feat_gens), val.to(device) # just omit the idxs entry