from rdkit import Chem
import dgl
import torch

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

    def append_reaction(self, react_smiles, prod_smiles): # one call -> one reaction loaded, and both params are lists
        react_canon = [Chem.CanonSmiles(sm) for sm in react_smiles]
        prod_canon = [Chem.CanonSmiles(sm) for sm in prod_smiles]

        self.r_p_canon.append((react_canon, prod_canon))

        for canon in react_canon + prod_canon:
            if canon in self.canon_to_mol:
                continue
            self.canon_to_mol[canon] = Chem.MolFromSmiles(canon)

### generate dgl graphs (non reaction specific) and mappings (reaction specific)
class DGLwRxnMappings:
    def __init__(self, directsmilesrepo):
        self.canon_to_dgl = None
        self.r_p_rxn_mappings = [] # order associated with entries of r_p_canon of entered DirectSmilesRepo

        self._load_from_directsmilesrepo(directsmilesrepo)
    
    def _load_from_directsmilesrepo(self, dsr: DirectSmilesRepo): # dsr - direct smiles repo because i hate typing
        self.canon_to_dgl = {canon : self.dgl_from_mol(mol) for canon, mol in dsr.canon_to_mol.items()}

        assert False, 'do your mappings loser, indivudally and as list too(?)'

    @staticmethod
    def dgl_from_mol(mol):
        from_a, to_b = [], []
        for b in mol.GetBonds():
            from_a.append(b.GetBeginAtomIdx())
            to_b.append(b.GetIdx())
            from_a.append(b.GetEndAtomIdx())
            to_b.append(b.GetIdx())
        
        a2b = (torch.tensor(from_a, dtype=torch.int), torch.tensor(to_b, dtype=torch.int))
        g2a = (torch.zeros(mol.GetNumAtoms(), dtype=torch.int), torch.arange(mol.GetNumAtoms(), dtype=torch.int))
        g2b = (torch.zeros(mol.GetNumBonds(), dtype=torch.int), torch.arange(mol.GetNumBonds(), dtype=torch.int))

        g = dgl.heterograph({ # reversed tuple are because this graph is bidirectional
            ('atom', 'a2b', 'bond') : a2b,
            ('bond', 'b2a', 'atom') : tuple(reversed(a2b)),
            ('global', 'g2a', 'atom') : g2a,
            ('atom', 'a2g', 'global') : tuple(reversed(g2a)),
            ('global', 'g2b', 'bond') : g2b,
            ('bond', 'b2g', 'global') : tuple(reversed(g2b)),
        })
        return g