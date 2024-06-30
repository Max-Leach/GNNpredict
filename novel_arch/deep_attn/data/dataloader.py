from torch.utils.data import DataLoader
import torch
import dgl
from copy import deepcopy

class RxnDataLoader(DataLoader):
    def __init__(self, rxndataset, **kwargs):
        super().__init__(rxndataset, collate_fn=self.collate_fn, **kwargs)
    
    def collate_fn(self, samples):
        data, values = [list(sub) for sub in zip(*samples)]
        _, _, rxn_feat_gens, idxs = [list(sub) for sub in zip(*data)] 
        # rxn_feat_gens = deepcopy(rxn_feat_gens) # clone here - feature generators
        r_p_graph_refs = [self.dataset.get_r_p_graph_ref(i) for i in idxs] 
        # r_p_graph_refs = deepcopy(r_p_graph_refs) # clone here to prevent dset from storing
        # load unique instances of graphs with idx
        ref_to_dgl = dict()
        ref_to_feats = {nt : dict() for nt in ['atom', 'bond', 'global']} # nt -> {ref -> feat}
        for r_p_ref in r_p_graph_refs:
            for refs in r_p_ref:
                for ref in refs:
                    if ref not in ref_to_dgl:
                        ref_to_dgl[ref] = self.dataset.get_dgl(ref)
                        for nt in ref_to_feats.keys():
                            ref_to_feats[nt][ref] = self.dataset.get_feats(nt, ref)
        # graphs and their feat loading
        graphs = tuple(ref_to_dgl.values()) 
        # graphs = deepcopy(graphs)# clone here to prevent dset graphs from storing features
        for nt in ref_to_feats:
            for g, ref in zip(graphs, ref_to_feats[nt]):
                g.nodes[nt].data['ft'] = ref_to_feats[nt][ref]
        batched_graph = dgl.batch(tuple(graphs))
        feats = {nt : batched_graph.nodes[nt].data['ft'] for nt in ['atom', 'bond', 'global']}
        feats = {nt : torch.tensor(self.dataset.transform[nt](feats[nt]), dtype=torch.float) for nt in feats}
        # map from r_p_graph_ref (entire dataset) idx to indices in graph batch above
        ref_to_local_idx = {ref : i for i, ref in enumerate(ref_to_dgl.keys())}
        for rxn_gen, r_p in zip(rxn_feat_gens, r_p_graph_refs):
            r_p = [[ref_to_local_idx[ref] for ref in reac_side] for reac_side in r_p]
            rxn_gen.reacs, rxn_gen.prods = r_p
        for r, ref in zip(rxn_feat_gens, r_p_graph_refs):
            r._final_graph = ref_to_local_idx[ref[0][0]]
        return (batched_graph, feats, rxn_feat_gens, idxs), torch.tensor(values)