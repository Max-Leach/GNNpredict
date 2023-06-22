from torch.utils.data import DataLoader
import torch
import dgl

class RxnDataLoader(DataLoader):
    def __init__(self, rxndataset, **kwargs):
        super().__init__(rxndataset, collate_fn=self.collate_fn, **kwargs)
    
    def collate_fn(self, samples):
        ## feat list
        # graphs, feats, self.rxn_feat_gens[idx], local_graph_ref, idx
        # _, _, rxn_feat_gens, idxs = [list(sub) for sub in zip(*samples)]
        data, values = [list(sub) for sub in zip(*samples)]
        _, _, rxn_feat_gens, idxs = [list(sub) for sub in zip(*data)]
        r_p_graph_refs = [self.dataset.get_r_p_graph_ref(i) for i in idxs]
        # load unique instances of graphs with idx
        ref_to_dgl = dict()
        ref_to_feats = {nt : dict() for nt in ['atom', 'bond', 'global']} # nt -> {ref -> feat}
        for r_p_ref in r_p_graph_refs:
            for refs in r_p_ref:
                for ref in refs:
                    if ref not in ref_to_dgl:
                        ref_to_dgl[ref] = self.dataset.dgl[ref]
                        for nt in ref_to_feats.keys():
                            ref_to_feats[nt][ref] = self.dataset.feats[nt][ref]
        graphs = tuple(ref_to_dgl.values())
        for nt in ref_to_feats:
            for g, ref in zip(graphs, ref_to_feats[nt]):
                g.nodes[nt].data['ft'] = ref_to_feats[nt][ref]
        batched_graph = dgl.batch(tuple(graphs))
        feats = {nt : batched_graph.nodes[nt].data['ft'] for nt in ['atom', 'bond', 'global']}
        feats = {nt : torch.tensor(self.dataset.transform[nt](feats[nt]), dtype=torch.float) for nt in feats}
        # map from r_p_graph_ref idx to indices in graph batch above
        ref_to_local_idx = {ref : i for i, ref in enumerate(ref_to_dgl.keys())}
        for rxn_gen, r_p in zip(rxn_feat_gens, r_p_graph_refs):
            r_p = [[ref_to_local_idx[ref] for ref in reac_side] for reac_side in r_p]
            rxn_gen.reacs, rxn_gen.prods = r_p
        # local_graph_refs = [tuple([[ref_to_local_idx[ref] for ref in reac_side] for reac_side in r_ps]) for r_ps in r_p_graph_refs]
        # NOTE: do feats! likely uses similar/merged proces with graphs
        return (batched_graph, feats, rxn_feat_gens, idxs), torch.tensor(values)