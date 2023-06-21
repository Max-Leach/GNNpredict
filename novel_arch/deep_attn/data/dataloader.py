from torch.utils.data import DataLoader
import dgl

class RxnDataLoader(DataLoader):
    def __init__(self, rxndataset, **kwargs):
        super().__init__(rxndataset, collate_fn=self.collate_fn, **kwargs)
    
    def collate_fn(self, samples):
        ## feat list
        # graphs, feats, self.rxn_feat_gens[idx], local_graph_ref, idx
        _, _, rxn_feat_gens, _, idxs = [list(sub) for sub in zip(*samples)]
        r_p_graph_refs = [self.dataset.get_r_p_graph_ref(i) for i in idxs]
        # load unique instances of graphs with idx
        ref_to_dgl = dict()
        for r_p_ref in r_p_graph_refs:
            for refs in r_p_ref:
                for ref in refs:
                    if ref not in ref_to_dgl:
                        ref_to_dgl[ref] = self.dataset.dgl[ref]
        graphs = dgl.batch(tuple(ref_to_dgl.values()))
        # map from r_p_graph_ref idx to indices in list above
        ref_to_local_idx = {ref : i for i, ref in enumerate(ref_to_dgl.keys())}
        local_graph_refs = [tuple([[ref_to_local_idx[ref] for ref in reac_side] for reac_side in r_ps]) for r_ps in r_p_graph_refs]
        # NOTE: do feats! likely uses similar/merged proces with graphs
        return graphs, None, rxn_feat_gens, local_graph_refs, idxs