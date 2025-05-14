from architecture.data.initial_containers import DGLwBDEMappings
import networkx.algorithms.isomorphism as nx_iso
from rdkit import Chem

def prod_and_react_nx_graphs(mols, broken_idx): # return nx graphs, first has product dgl graph indices, second has indices of seperated reactant
    r_mol, p_mols = mols

    p_frags = Chem.GetMolFrags(Chem.FragmentOnBonds(r_mol, broken_idx))
    r_nx = DGLwBDEMappings.mol_to_nx(r_mol)
    p_nxs = [r_nx.subgraph(p) for p in p_frags]
    p_mol_nxs = [DGLwBDEMappings.mol_to_nx(m) for m in p_mols]
    return p_mol_nxs, p_nxs
    
def prod_to_reac_atom_map(mols, broken_idx): # assumes single reactant, single bond broken, two products, find mapping from prod graph to reactant given mols
    p_mol_nxs, p_nxs = prod_and_react_nx_graphs(mols, broken_idx)

    if len(p_mol_nxs) < 2: # did not properly match
        return []

    nm = nx_iso.categorical_node_match('specie', 'BAD_MATCH')
    # assumes two products from here
    # find which product maps to which product in split molecule
    pnx_checks = set([0, 1])
    pmol_0, pmol_1 = p_mol_nxs
    pnx_match_first = None
    first_map = None
    for s in pnx_checks:
        match = nx_iso.GraphMatcher(pmol_0.to_undirected(), p_nxs[s].to_undirected(), nm)
        if match.is_isomorphic():
            pnx_match_first = s
            first_map = match.mapping
            break
    # check if one member left in set, otherwise something is wrong with assumption of 1 reactant -> 2 products
    assert type(pnx_match_first) is int and first_map != None, 'No matching for first product!'
    pnx_checks.remove(pnx_match_first)
    pnx_match_sec = next(iter(pnx_checks))
    sec_match = nx_iso.GraphMatcher(pmol_1.to_undirected(), p_nxs[pnx_match_sec].to_undirected(), nm)
    assert sec_match.is_isomorphic(), 'No match for second product!: {}'.format(reac_idx)
    sec_map = sec_match.mapping
    # map from concat product features list to reactant features
    return [first_map, sec_map]

def prod_to_reac_bond_map(prod_mols, prods_atom_map, r_mol, broken_bond_idx): # single reaction, generate bond mapping
    prods = prod_mols
    prod_bond_to_patoms = [{b.GetIdx() : (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in p.GetBonds()} for p in prods]
    prod_bond_to_ratoms = [{b: tuple(sorted([atom_map[a] for a in bmap[b]])) for b in bmap} for atom_map, bmap in zip(prods_atom_map, prod_bond_to_patoms)]
    reac_atoms_to_bond = {tuple(sorted([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])) : b.GetIdx() for b in r_mol.GetBonds()}
    pbond_to_rbond = [{b: reac_atoms_to_bond[bmap[b]] for b in bmap} for bmap in prod_bond_to_ratoms]
    # insert broken bond mapping, at end of product array assumed
    pbond_to_rbond.append({0: broken_bond_idx})
    return pbond_to_rbond