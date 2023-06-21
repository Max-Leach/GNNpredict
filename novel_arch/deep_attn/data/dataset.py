from torch.utils.data import Dataset
from novel_arch.deep_attn.data.initial_containers import DirectSmilesRepo, DGLwBDEMappings
from novel_arch.deep_attn.rxn_graph import BondDissociate

# initial containers of bde dataset into entries
class BDEDataset(Dataset):
    def __init__(self, dsr: DirectSmilesRepo, bdemap: DGLwBDEMappings, load_graphs=False):
        super().__init__()
        self._organize_data(dsr, bdemap)
        self.load_graphs = load_graphs # don't load graphs directly, such as in case where dataloader handles it
    
    # data features not handled by simply copying as above will be organized into more efficient form here
    def _organize_data(self, dsr: DirectSmilesRepo, bdemap: DGLwBDEMappings):
        self.dgl = list(bdemap.canon_to_dgl.values())
        canon_to_idx = {c : idx for idx, c in enumerate(bdemap.canon_to_dgl.keys())}
        self.r_p_graph_ref = [([canon_to_idx[r] for r in rs], [canon_to_idx[p] for p in ps]) for rs, ps in dsr.r_p_canon]
        reac_graphs = [self.dgl[rs[0]] for rs, _ in self.r_p_graph_ref]
        bdegen_properties = zip(bdemap.rxn_atom_mappings, bdemap.rxn_bond_mappings, bdemap.prods_has_bonds, reac_graphs)
        self.rxn_feat_gens = [BondDissociate(am, bm, p_has_bs, final_g) for am, bm, p_has_bs, final_g in bdegen_properties]

    def get_r_p_graph_ref(self, idx):
        return self.r_p_graph_ref[idx]

    def __getitem__(self, idx):
        # may change graph aggregation st graphs are pulled from the full graph list and ref idxs are generated on the fly
        if self.load_graphs:
            graph_refs = [rp for rplist in self.r_p_graph_ref[idx] for rp in rplist]
            graphs = [self.dgl[i] for i in graph_refs]
            # NOTE: also do feats here
            feats = None
            local_graph_ref = ([0], [1, 2]) # doesn't change due to fixed way graphs are aggregated above
        else:
            graphs = None
            feats = None
            local_graph_ref = None
        return graphs, feats, self.rxn_feat_gens[idx], local_graph_ref, idx
    
    def __len__(self):
        return len(self.r_p_graph_ref)