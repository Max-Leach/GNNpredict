from rdkit import Chem
import dgl
import torch
import networkx.algorithms.isomorphism as nx_iso
import networkx as nx

''' 
    end storage should help create at end: 
    all unique molecules stored, 
    associated features for all, 
    cover items for BondDissociate instance:
        mappings b/w reactant, products features concated

    final dataset should batch required graphs, get batched features, and return rxngenerator that point to batched properly
'''

### first step from raw smiles, generate canon -> mol and reactions with their canon mols
class DirectSmilesRepo:
    def __init__(self):
        # following two should have same keys
        self.canon_to_mol = dict() # canonical smiles -> rdkit mol
        self.r_p_canon = [] # 1 ([reactant canon smiles], [product canon smiles]) for each reaction
        self.react_broken_bonds = [] # broken bonds list for each reaction, may change to a generic auxillary property object later if going beyond BDEs
        self.canon_to_atom_bond_map = dict() # canonical smiles -> {atom pair -> bond}

    def append_reaction(self, react_smiles, prod_smiles, broken_bond_idxs): # one call -> one reaction loaded, and both params are lists
        react_canon = [Chem.CanonSmiles(sm) for sm in react_smiles]
        prod_canon = [Chem.CanonSmiles(sm) for sm in prod_smiles]

        self.r_p_canon.append((react_canon, prod_canon))
        self.react_broken_bonds.append(broken_bond_idxs)

        for canon in react_canon + prod_canon:
            if canon in self.canon_to_mol:
                continue
            mol = Chem.AddHs(Chem.MolFromSmiles(canon))
            self.canon_to_mol[canon] = mol
            self.canon_to_atom_bond_map[canon] = {(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) : b.GetIdx() for b in mol.GetBonds()}

### generate dgl graphs (non reaction specific) and mappings (reaction specific, atom and bond)
class DGLwBDEMappings:
    def __init__(self, directsmilesrepo):
        self.canon_to_dgl = None
        self.rxn_atom_mappings = None # order associated with entries of r_p_canon of entered DirectSmilesRepo
        self.rxn_bond_mappings = None # same as above

        # self.nm = nx_iso.categorical_node_match('specie', 'BAD_MATCH')
        self._load_from_directsmilesrepo(directsmilesrepo)
    
    def _load_from_directsmilesrepo(self, dsr: DirectSmilesRepo): # dsr - direct smiles repo because i hate typing
        self.canon_to_dgl = {canon : DGLwBDEMappings.dgl_from_mol(mol) for canon, mol in dsr.canon_to_mol.items()}

        atom_map_for_rxn = [DGLwBDEMappings.prod_to_reac_atom_map(i, dsr) for i in range(len(dsr.r_p_canon))]
        self.rxn_atom_mappings = [DGLwBDEMappings.to_concat_map(m) for m in atom_map_for_rxn]
        bond_map_for_rxn = [DGLwBDEMappings.prod_to_reac_bond_map(i, at_map, dsr) for i, at_map in enumerate(atom_map_for_rxn)]
        self.rxn_bond_mappings = [DGLwBDEMappings.to_concat_map(m) for m in bond_map_for_rxn]

    @staticmethod
    def prod_to_reac_atom_map(reac_idx, dsr: DirectSmilesRepo): # assumes single reactant, single bond broken, two products, find mapping from prod graph to reactant given reaction idx in dsr
        p_mol_nxs, p_nxs = DGLwBDEMappings.prod_and_react_nx_graphs(reac_idx, dsr)
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
        assert sec_match.is_isomorphic(), 'No match for second product!'
        sec_map = sec_match.mapping
        # map from concat product features list to reactant features
        return [first_map, sec_map]
        # return [list(m) for m in [first_map.items(), sec_map.items()]]
    
    @staticmethod
    def prod_to_reac_bond_map(reac_idx, prods_atom_map, dsr: DirectSmilesRepo): # single reaction, generate bond mapping
        reacs, prods = dsr.r_p_canon[reac_idx]
        reac_sm = reacs[0] # single reactant assumption
        prods = [dsr.canon_to_mol[c] for c in prods]
        prod_bond_to_patoms = [{b.GetIdx() : (b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in p.GetBonds()} for p in prods] # work from here!
        prod_bond_to_ratoms = [{b: tuple(sorted([atom_map[a] for a in bmap[b]])) for b in bmap} for atom_map, bmap in zip(prods_atom_map, prod_bond_to_patoms)]
        reac_atoms_to_bond = dsr.canon_to_atom_bond_map[reac_sm]
        pbond_to_rbond = [{b: reac_atoms_to_bond[bmap[b]] for b in bmap} for bmap in prod_bond_to_ratoms]
        # insert broken bond mapping, at end of product array assumed
        broken_bond = dsr.react_broken_bonds[reac_idx][0]
        pbond_to_rbond.append({0: broken_bond})
        return pbond_to_rbond

    @staticmethod
    def to_concat_map(maps_for_prod):
        map_idx_offset = [0]
        for m in maps_for_prod[:-1]:
            map_idx_offset.append(map_idx_offset[-1] + len(m))
        concat_map = []
        for m_for_prod, offset in zip(maps_for_prod, map_idx_offset):
            for k in m_for_prod:
                # p, r_key = m
                new_m = (k + offset, m_for_prod[k])
                concat_map.append(new_m)
        return [e[0] for e in sorted(concat_map, key=lambda e: e[1])]

    @staticmethod
    def prod_and_react_nx_graphs(reac_idx, dsr): # return nx graphs, first has product dgl graph indices, second has indices of seperated reactant
        reacs, prods = dsr.r_p_canon[reac_idx]
        reac = reacs[0] # single reactant assumption
        r_mol = dsr.canon_to_mol[reac]
        broken_idx = dsr.react_broken_bonds[reac_idx]
        p_frags = Chem.GetMolFrags(Chem.FragmentOnBonds(r_mol, broken_idx))
        r_nx = DGLwBDEMappings.mol_to_nx(r_mol)
        p_nxs = [r_nx.subgraph(p) for p in p_frags]
        p_mols = [dsr.canon_to_mol[canon] for canon in prods]
        p_mol_nxs = [DGLwBDEMappings.mol_to_nx(m) for m in p_mols]
        return p_mol_nxs, p_nxs

    @staticmethod
    def mol_to_nx(mol):
        ng = nx.Graph()
        ng.add_nodes_from(range(mol.GetNumAtoms()))
        for b in mol.GetBonds():
            ng.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
        for a in mol.GetAtoms():
            ng.nodes[a.GetIdx()]['specie'] = a.GetAtomicNum()
        return ng

    @staticmethod
    def dgl_from_mol(mol):
        from_a, to_b = [], []
        for b in mol.GetBonds():
            from_a.append(b.GetBeginAtomIdx())
            to_b.append(b.GetIdx())
            from_a.append(b.GetEndAtomIdx())
            to_b.append(b.GetIdx())
        if mol.GetNumAtoms() == 1: # create fake bond so graph processing will work, from the single atom
            assert mol.GetBonds() == 0, 'how do you have one atom but non zero bonds???'
            from_a.append(0)
            to_b.append(0)
        
        a2b = (torch.tensor(from_a, dtype=torch.int), torch.tensor(to_b, dtype=torch.int))
        # only one global node, so use zeros to point from single global node to all other nodes
        g2a = (torch.zeros(mol.GetNumAtoms(), dtype=torch.int), torch.arange(mol.GetNumAtoms(), dtype=torch.int))
        g2b = (torch.zeros(mol.GetNumBonds(), dtype=torch.int), torch.arange(mol.GetNumBonds(), dtype=torch.int))

        g = dgl.heterograph({ # reversed tuple for many because this graph is bidirectional
            ('atom', 'a2b', 'bond') : a2b,
            ('bond', 'b2a', 'atom') : tuple(reversed(a2b)),
            ('global', 'g2a', 'atom') : g2a,
            ('atom', 'a2g', 'global') : tuple(reversed(g2a)),
            ('global', 'g2b', 'bond') : g2b,
            ('bond', 'b2g', 'global') : tuple(reversed(g2b)),
        })
        # NOTE: may remove - load atomic number as feature
        specie_for_atom = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()])
        g.nodes['atom'].data.update({'specie' : specie_for_atom})
        species_for_bond = torch.tensor([[b.GetBeginAtom().GetAtomicNum(), b.GetEndAtom().GetAtomicNum()] for b in mol.GetBonds()])
        g.nodes['bond'].data.update({'species' : species_for_bond})
        return g