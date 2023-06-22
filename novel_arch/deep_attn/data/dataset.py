from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from novel_arch.deep_attn.data.initial_containers import DirectSmilesRepo, DGLwBDEMappings
from novel_arch.deep_attn.rxn_graph import BondDissociate

# interpet array of inhomogenous data to normalize across certain column in NP
# (such as graph features, where node count vary, but you want to normalize across entire dataset for each node feat column)
# no copying *should* occur
class InHomoInterpretNP:
    def __init__(self, inhomo_seq):
        self.seq = inhomo_seq
        self.idx_to_inhomo_ref = []
        for i in range(len(inhomo_seq)):
            item = inhomo_seq[i]
            for j in range(len(item)):
                self.idx_to_inhomo_ref.append((i, j))

    def __getitem__(self, idx):
        par, sub = self.idx_to_inhomo_ref[idx]
        return self.seq[par][sub].numpy()

    def __len__(self):
        return len(self.idx_to_inhomo_ref)

# initial containers of bde dataset into entries
class BDEDataset(Dataset):
    # featurizers - dict of callables under 'atom', 'bond', 'global' keys
    def __init__(self, dsr: DirectSmilesRepo, bdemap: DGLwBDEMappings, featurizers, std_data=True, load_graphs=False):
        super().__init__()
        self.std_data = std_data
        self.load_graphs = load_graphs # don't load graphs directly, such as in case where dataloader handles it
        self._fill_data(dsr, bdemap, featurizers)
    
    # data features not handled by simply copying as above will be organized into more efficient form here
    def _fill_data(self, dsr: DirectSmilesRepo, bdemap: DGLwBDEMappings, featurizers):
        self.values = dsr.values
        self._graph_data(dsr, bdemap, featurizers)
        canon_to_idx = {c : idx for idx, c in enumerate(bdemap.canon_to_dgl.keys())}
        self.r_p_graph_ref = [([canon_to_idx[r] for r in rs], [canon_to_idx[p] for p in ps]) for rs, ps in dsr.r_p_canon]
        reac_graphs = [self.dgl[rs[0]] for rs, _ in self.r_p_graph_ref]
        bdegen_properties = zip(bdemap.rxn_atom_mappings, bdemap.rxn_bond_mappings, bdemap.prods_has_bonds, reac_graphs)
        self.rxn_feat_gens = [BondDissociate(am, bm, p_has_bs, final_g) for am, bm, p_has_bs, final_g in bdegen_properties]

    # create dgl graphs and features associated with them
    def _graph_data(self, dsr: DirectSmilesRepo, bdemap: DGLwBDEMappings, featurizers):
        self.dgl = list(bdemap.canon_to_dgl.values())
        # assume featurizers has 'atom', 'bond', 'global' keys
        # use canon_to_dgl for values to preserve ordering b/w that and self.dgl created above
        self.feats = {nt : [featurizers[nt](dsr.canon_to_mol[c]) for c in bdemap.canon_to_dgl.keys()] for nt in featurizers}
        self._std_setup()
    
    def _std_setup(self):
        self.stders = dict()
        self.transform = dict()
        for nt in self.feats:
            if self.std_data:
                scaler = StandardScaler()
                feats = InHomoInterpretNP(self.feats[nt])
                scaler.fit_transform(feats)
                self.stders[nt] = scaler
                self.transform[nt] = self.stders[nt].transform
            else:
                self.transform[nt] = lambda x: x

    def get_r_p_graph_ref(self, idx):
        return self.r_p_graph_ref[idx]

    def __getitem__(self, idx):
        # may change graph aggregation st graphs are pulled from the full graph list and ref idxs are generated on the fly
        if self.load_graphs:
            graph_refs = [rp for rplist in self.r_p_graph_ref[idx] for rp in rplist]
            graphs = [self.dgl[i] for i in graph_refs]
            feats = {nt : self.transform[nt](self.feats[nt][idx]) for nt in self.feats}
            self.rxn_feat_gens[idx].reacs, self.rxn_feat_gens[idx].prods = [0], [1, 2] # doesn't change due to fixed way graphs are aggregated above
        else:
            graphs = None
            feats = None
        return (graphs, feats, self.rxn_feat_gens[idx], idx), self.values[idx]
    
    def __len__(self):
        return len(self.r_p_graph_ref)