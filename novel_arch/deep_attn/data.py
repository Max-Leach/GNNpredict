from rdkit import Chem
import dgl
import torch
import networkx.algorithms.isomorphism as nx_iso

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

    def append_reaction(self, react_smiles, prod_smiles, broken_bond_idxs): # one call -> one reaction loaded, and both params are lists
        react_canon = [Chem.CanonSmiles(sm) for sm in react_smiles]
        prod_canon = [Chem.CanonSmiles(sm) for sm in prod_smiles]

        self.r_p_canon.append((react_canon, prod_canon))
        self.react_broken_bonds.append(broken_bond_idxs)

        for canon in react_canon + prod_canon:
            if canon in self.canon_to_mol:
                continue
            self.canon_to_mol[canon] = Chem.AddHs(Chem.MolFromSmiles(canon))

### generate dgl graphs (non reaction specific) and mappings (reaction specific, atom and bond)
class DGLwBDEMappings:
    def __init__(self, directsmilesrepo):
        self.canon_to_dgl = None
        self.r_p_rxn_mappings = [] # order associated with entries of r_p_canon of entered DirectSmilesRepo

        self.nm = nx_iso.categorical_node_match('specie', 'BAD_MATCH')
        self._load_from_directsmilesrepo(directsmilesrepo)
    
    def _load_from_directsmilesrepo(self, dsr: DirectSmilesRepo): # dsr - direct smiles repo because i hate typing
        self.canon_to_dgl = {canon : DGLwBDEMappings.dgl_from_mol(mol) for canon, mol in dsr.canon_to_mol.items()}

        assert False, 'do your mappings loser, indivudally and as list too(?)'

    @staticmethod
    def prod_to_reac_atom_map(reac_idx, dsr: DirectSmilesRepo): # assumes single reactant, single bond broken, multiple products, find mapping from prod graph to reactant given reaction idx in dsr
        reacs, prods = dsr.r_p_canon[reac_idx]
        reac = reacs[0] # single reactant assumption
        r_mol = dsr.cannon_to_mol[reac]
        broken_idx = dsr.react_broken_bonds[reac_idx]
        p_frags = Chem.GetMolFrags(Chem.FragmentOnBonds(r_mol, broken_idx))
        r_nx = DGLwBDEMappings.mol_to_nx(r_mol)
        p_nxs = [r_nx.subgraph(p) for p in p_frags]
        mols = [dsr.canon_to_mol[canon] for canon in prods]
        mol_nxs = [DGLwBDEMappings.mol_to_nx(m) for m in mols]
        '''
            hi this is me from the future, you suck

            use that trash test you made yesterday to map between atoms on fragments (using p_nxs) and 
                products generated without fragmenting (mol_nxs), with GraphMatcher
            \/\/
            with atom mappings, generate bond mappings using pairs, likely put in a seperate function
        '''
        # NOTE: WORK FROM HERE!!

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
        return g