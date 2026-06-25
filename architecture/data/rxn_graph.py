import dgl
import torch
from abc import abstractmethod
from rdkit import Chem
import networkx as nx
import numpy as np
from architecture.data.nsppk import NodeNSPPK
from scipy.sparse import vstack, csr_matrix

# generic reaction, accept list of reactants and products, return features of a larger graph
class RxnFeatGenerator:
    @abstractmethod
    def get_g_fts(self, reac_feats, prod_feats):
        pass

    @abstractmethod
    def get_rxn_graph(self):
        pass

# bond breaking reaction
class BondDissociate(RxnFeatGenerator):
    def __init__(self, atom_mapping, bond_mapping, prods_has_bond, final_graph):
        # prods_has_bonds - list of bool for each product if it has bonds
        self.reacs, self.prods = None, None  # reference graphs in larger batch
        self.mappings = {
            'global': [0],
            'atom': atom_mapping,
            'bond': bond_mapping,
        }
        self.prods_has_bond = prods_has_bond
        self._final_graph = final_graph

    def get_rxn_graph(self):
        return self._final_graph

    def get_g_fts(self, reac_feats, prod_feats):
        prd_feats_updated = {}
        reac_feats_updated = {}
        for nt in ['atom', 'global']:
            prd_feats_updated[nt] = torch.cat([ft[nt] for ft in prod_feats])
        for nt in ['bond', 'atom', 'global']:
            reac_feats_updated[nt] = torch.cat([ft[nt] for ft in reac_feats])
        # remove fictional bond cases - if no bond mapping there are no real bonds
        prod_bond_feats = [prod_feats[p]['bond'] for p in
                           filter(lambda i: self.prods_has_bond[i], range(len(self.prods_has_bond)))]
        # add bond to end to represent broken bond
        prod_bond_feats.append(torch.zeros_like(prod_bond_feats[0]))
        prd_feats_updated['bond'] = torch.cat(prod_bond_feats)

        prd_feats_updated['global'] = torch.sum(prd_feats_updated['global'], dim=0, keepdim=True)
        reac_feats_updated['global'] = torch.sum(reac_feats_updated['global'], dim=0, keepdim=True)

        final_feats = {}
        for nt in self.mappings.keys():
            final_feats[nt] = prd_feats_updated[nt][self.mappings[nt]] - reac_feats_updated[nt]

        return final_feats

# keep reaction graph tensors seperate
def get_rxn_feat_list(batched_graph, batched_feats, rxns):
    for nt in batched_feats:
        batched_graph.nodes[nt].data.update({'ft': batched_feats[nt]})

    graphs = dgl.unbatch(batched_graph)
    graph_feats = [{nt: g.nodes[nt].data['ft'] for nt in batched_feats} for g in graphs]
    rxn_in_feats = [{'reac': [graph_feats[r] for r in rxn.reacs], 'prod': [graph_feats[p] for p in rxn.prods]} for rxn
                    in rxns]
    # for r in rxns:
    #     r._final_graph = graphs[r._final_graph]
    rxn_feat_gens = rxns
    return [gen.get_g_fts(rxn_in_feat['reac'], rxn_in_feat['prod']) for gen, rxn_in_feat in
            zip(rxn_feat_gens, rxn_in_feats)]
