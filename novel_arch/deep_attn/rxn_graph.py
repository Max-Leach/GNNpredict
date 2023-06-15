import dgl
import torch
from abc import abstractmethod

# reactants, products indexint
# atom, bond mapping (global index is guaranteed to be same, only one)

# also need to figure out what to do with destroyed bonds

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
        prod_bond_feats = [prod_feats[p]['bond'] for p in filter(lambda i: self.prods_has_bond[i], range(len(self.prods_has_bond)))]
        # add bond to end to represent broken bond
        prod_bond_feats.append(torch.zeros_like(prod_bond_feats[0]))
        prd_feats_updated['bond'] = torch.cat(prod_bond_feats)

        prd_feats_updated['global'] = torch.sum(prd_feats_updated['global'], dim=0, keepdim=True)
        reac_feats_updated['global'] = torch.sum(reac_feats_updated['global'], dim=0, keepdim=True)

        final_feats = {}
        for nt in self.mappings.keys():
            final_feats[nt] = prd_feats_updated[nt][self.mappings[nt]] - reac_feats_updated[nt]

        return final_feats

# bondnet-style reaction list + all features to single reaction graph
# precursor to both: reaction graph production in actual model; preprocessing the dataset
def bondnet_batch_to_own(batched_graph, batched_feats, reactions):
    for nt in batched_feats:
        batched_graph.nodes[nt].data.update({'ft': batched_feats[nt]})
    # print('nombre d\'atom', batched_graph.num_nodes('atom'))
    
    graphs = dgl.unbatch(batched_graph)
    graph_feats = [{nt: g.nodes[nt].data['ft'] for nt in batched_feats} for g in graphs]
    rxn_in_feats = [{'reac' : [graph_feats[r] for r in rxn.reactants], 'prod' : [graph_feats[p] for p in rxn.products]} for rxn in reactions]
    rxn_feat_gens = [BondDissociate(rxn.atom_mapping_as_list, rxn.bond_mapping_as_list, tuple(map(lambda m: len(m) > 0, rxn.bond_mapping)), graphs[rxn.reactants[0]]) for rxn in reactions]
    
    # arond this point will be where reactions processing will remain same for our set

    rxns_feats_list = [gen.get_g_fts(rxn_in_feat['reac'], rxn_in_feat['prod']) for gen, rxn_in_feat in zip(rxn_feat_gens, rxn_in_feats)]

    rxns_feats = {}
    for nt in batched_feats.keys():
        rxns_feats[nt] = torch.cat([ft[nt] for ft in rxns_feats_list])

    batched_graph = dgl.batch([gen.get_rxn_graph() for gen in rxn_feat_gens])

    return rxns_feats, batched_graph