from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from statistics import mean, stdev
import dgl
from os import path as opath
import os
import pickle
import time
import heapq
import torch

from novel_arch.deep_attn.data.initial_containers import DirectSmilesRepo, DGLwBDEMappings
from novel_arch.deep_attn.data.rxn_graph import BondDissociate

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
    @staticmethod
    def from_initials(dsr: DirectSmilesRepo, bdemap: DGLwBDEMappings, featurizers, std_data=True, load_graphs=False):
        dset = BDEDataset()
        dset.std_data = std_data
        dset.load_graphs = load_graphs # don't load graphs directly, such as in case where dataloader handles it
        # dset.bdemap = bdemap # store which makes saving easier
        dset._fill_data(dsr, bdemap, featurizers)
        return dset
    
    @staticmethod
    def from_subset(subset):
        indices = subset.indices
        dset = subset.dataset

        r_p_graph_ref = [dset.r_p_graph_ref[i] for i in indices]
        g_idxs = {gi for rplist in r_p_graph_ref for gilist in rplist for gi in gilist}
        g_idxs = tuple(g_idxs)
        orig_to_new = {gi : i for i, gi in enumerate(g_idxs)}

        new_set = BDEDataset()
        new_set.values = [dset.values[i] for i in indices]
        new_set.dgl = [dset.dgl[gi] for gi in g_idxs]
        new_set.feats = {nt : [dset.feats[nt][gi] for gi in g_idxs] for nt in dset.feats}
        new_set.r_p_graph_ref = [[[orig_to_new[rp] for rp in rpsub] for rpsub in rplist] for rplist in r_p_graph_ref]
        new_set.rxn_feat_gens = [dset.rxn_feat_gens[i] for i in indices]
        new_set.load_graphs = dset.load_graphs
        new_set.std_data = dset.std_data
        new_set._std_setup()
        new_set._value_mean_stdev()

        return new_set
    
    @staticmethod
    def load(path):
        as_lazy = opath.exists(opath.join(path, 'dgl_dir'))
        dset = BDEDataset()
        if as_lazy:
            dset.path = path
            dset.dgl_cache = dict()
            dset.dgl_access_time = dict()
            dset.feat_cache = dict()
            dset.feat_access_time = dict ()
        else:
            dset.dgl, _ = dgl.load_graphs(opath.join(path, 'dgl'))
        with open(opath.join(path, 'picklable'), 'rb') as f:
            block = pickle.load(f)
            (dset.rxn_feat_gens,
                    dset.values, 
                    feats, 
                    dset.r_p_graph_ref, 
                    dset.stders, 
                    dset.transform, 
                    dset.val_mean, 
                    dset.val_stdev,
                    dset.load_graphs,
                    dset.std_data, 
                    ) = block
            if not as_lazy:
                dset.feats = feats
                dset._populate_reac_graph()
        return dset

    def save(self, path, as_lazy=False):
        if as_lazy:
            try:
                self.path
                raise Exception('can\'t save dataset thats already lazy!')
            except AttributeError:
                pass
        path = opath.join(path, 'dset')
        if not opath.exists(path):
            os.makedirs(path)
        feats_holder = None
        if as_lazy:
            dgls_path = opath.join(path, 'dgl_dir')
            os.makedirs(dgls_path)
            for i, g in enumerate(self.dgl):
                dgl.save_graphs(opath.join(dgls_path, str(i)), g)
            feats_path = opath.join(path, 'feat_dir')
            os.makedirs(feats_path)
            for nt in self.feats:
                nt_path = opath.join(feats_path, nt)
                os.makedirs(nt_path)
                for i, ft in enumerate(self.feats[nt]):
                    with open(opath.join(nt_path, str(i)), 'wb+') as f:
                        torch.save(ft, f)
        else:
            feats_holder = self.feats
            dgl.save_graphs(opath.join(path, 'dgl'), self.dgl)
        feat_gens_no_graph = [BondDissociate(fg.mappings['atom'], fg.mappings['bond'], fg.prods_has_bond, None) for fg in self.rxn_feat_gens]
        block = (feat_gens_no_graph,
                    self.values, 
                    feats_holder, 
                    self.r_p_graph_ref, 
                    self.stders, 
                    self.transform, 
                    self.val_mean, 
                    self.val_stdev,
                    self.load_graphs,
                    self.std_data,)
        with open(opath.join(path, 'picklable'), 'wb') as f:
                pickle.dump(block, f)

    # data features not handled by simply copying as above will be organized into more efficient form here
    def _fill_data(self, dsr: DirectSmilesRepo, bdemap: DGLwBDEMappings, featurizers):
        self.values = dsr.values
        self._value_mean_stdev()
        self._graph_data(dsr, bdemap, featurizers)
        self._rxn_specific_data(dsr, bdemap)

    # load indices into graph/features list, create feature generators - both for each reaction
    def _rxn_specific_data(self, dsr: DirectSmilesRepo, bdemap: DGLwBDEMappings):
        canon_to_idx = {c : idx for idx, c in enumerate(bdemap.canon_to_dgl.keys())}
        self.r_p_graph_ref = [([canon_to_idx[r] for r in rs], [canon_to_idx[p] for p in ps]) for rs, ps in dsr.r_p_canon]
        self._rxn_gen_load(bdemap)

    def _rxn_gen_load(self, bdemap):
        reac_graphs = [self.get_dgl(rs[0]) for rs, _ in self.r_p_graph_ref]
        bdegen_properties = zip(bdemap.rxn_atom_mappings, bdemap.rxn_bond_mappings, bdemap.prods_has_bonds, reac_graphs)
        self.rxn_feat_gens = [BondDissociate(am, bm, p_has_bs, final_g) for am, bm, p_has_bs, final_g in bdegen_properties]
    
    def _populate_reac_graph(self):
        # reac_graphs = [self.get_dgl(rs[0]) for rs, _ in self.r_p_graph_ref]
        # for rg, feat_gen in zip(reac_graphs, self.rxn_feat_gens):
        #     feat_gen._final_graph = rg
        reac_graphs = [self._get_complete_rxn_feat_gen(i) for i in range(len(self.rxn_feat_gens))]
    
    def _get_complete_rxn_feat_gen(self, idx):
        rxn_feat_gen = self.rxn_feat_gens[idx]
        rs, _ = self.r_p_graph_ref[idx]
        rxn_feat_gen._final_graph = self.get_dgl(rs[0])
        return rxn_feat_gen

    def _value_mean_stdev(self):
        self.val_mean = mean(self.values)
        self.val_stdev = stdev(self.values)

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

    def _cache_access(self, cache, access_time, cant_find_item_fn, k):
        if len(cache) > 5000:
            new_time = {e : access_time[e] for e in heapq.nlargest(1000, cache.keys(), key=lambda e: access_time[e])}
            new_cache = {e : cache[e] for e in new_time.keys()}
            cache = new_cache
            access_time = new_time
        
        if k not in cache:
            cache[k] = cant_find_item_fn(k)
        access_time[k] = time.perf_counter()

        return cache[k], cache, access_time

    def get_dgl(self, i):
        try:
            def open_dgl(i): # this throws attributeerror if no lazy loading
                gpath = opath.join(self.path, 'dgl_dir', str(i)) # do this first to check if this is lazy loaded
                return dgl.load_graphs(gpath)[0][0]
            g, self.dgl_cache, self.dgl_access_time = self._cache_access(self.dgl_cache, self.dgl_access_time, open_dgl, i)
        except AttributeError: # see above on check
            g = self.dgl[i]
        return g
    
    def get_feats(self, nt, i):
        try:
            def open_feat(k): # this throws attributeerror if no lazy loading
                nt, i = k
                fpath = opath.join(self.path, 'feat_dir', nt, str(i))
                with open(fpath, 'rb') as f:
                    return torch.load(f)
            f, self.feat_cache, self.feat_access_time = self._cache_access(self.feat_cache, self.feat_access_time, open_feat, (nt, i))
        except AttributeError: # see above on check
            f = self.feats[nt][i]
        return f

    def __getitem__(self, idx):
        # may change graph aggregation st graphs are pulled from the full graph list and ref idxs are generated on the fly
        if self.load_graphs:
            graph_refs = [rp for rplist in self.r_p_graph_ref[idx] for rp in rplist]
            graphs = [self.get_dgl(i) for i in graph_refs]
            # VVVV this may be wrong!
            # feats = {nt : self.transform[nt](self.get_feats(nt, idx)) for nt in self.feats}
            feats = {nt : self.transform[nt](torch.cat([self.get_feats(nt, r) for r in graph_refs])) for nt in ['bond', 'atom', 'global']}
            self.rxn_feat_gens[idx].reacs, self.rxn_feat_gens[idx].prods = [0], [1, 2] # doesn't change due to fixed way graphs are aggregated above
        else:
            graphs = None
            feats = None
        return (graphs, feats, self._get_complete_rxn_feat_gen(idx), idx), self.values[idx]
    
    def __len__(self):
        return len(self.r_p_graph_ref)


## subset of above, plug in main dataset and indices that will be here
# important for the loader used to aggregated graphs
class BDESubset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        (graph, feat, gen, _), val = self.dataset[self.indices[idx]]
        return (graph, feat, gen, idx), val

    def get_r_p_graph_ref(self, idx):
        return self.dataset.get_r_p_graph_ref(self.indices[idx])

    @property
    def transform(self):
        return self.dataset.transform
    
    def get_dgl(self, i):
        return self.dataset.get_dgl(i)

    def get_feats(self, nt, i):
        return self.dataset.get_feats(nt, i)
    
    @property
    def val_stdev(self):
        return self.dataset.val_stdev

    @property
    def val_mean(self):
        return self.dataset.val_mean