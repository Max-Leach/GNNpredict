from novel_arch.deep_attn.model import DeepAtom
from novel_arch.deep_attn.data.dataloader import DataLoader

from novel_arch.deep_attn.data.initial_containers import DirectSmilesRepo, DGLwBDEMappings
from novel_arch.deep_attn.data.dataset import BDEDataset

dsr = DirectSmilesRepo()
dsr.append_reaction(['NCCCC(=O)O'], ['[CH2]CCC(=O)O', '[NH2]'], [0])
dsr.append_reaction(['NCCCC(=O)O'], ['[CH2]CC(=O)O', '[CH2]N'], [1])

dsr.append_reaction(['CN(N)CCCC(C)(C)C'], ['[CH2]CC(C)(C)C', '[CH2]N(C)N'], [3])
dsr.append_reaction(['COC(=O)OCCC#N'], ['CO[C]=O', 'N#CCC[O]'], [3])
dsr.append_reaction(['C/C=C/[C@@H](C)CCC#N'], ['[H]', 'C/C=C/[C@@H](C)[CH]CC#N'], [17])

bdemap = DGLwBDEMappings(dsr)
from novel_arch.deep_attn.data.featurizers import AtomFeaturize, BondFeaturize, GlobalFeaturize
aprop = ['atomic_num', 'total_degree', 'total_num_hs', 'ring_of_size', 'is_in_ring']
bprop = ['is_in_ring', 'ring_of_size', 'dative']
gprop = ['num_atoms', 'num_bonds', 'total_weight']
dset = BDEDataset(dsr, bdemap, featurizers={'atom' : AtomFeaturize(aprop, [1, 6, 7, 8]), 'bond' : BondFeaturize(bprop), 'global' : GlobalFeaturize(gprop),}, load_graphs=False)

from novel_arch.deep_attn.data.dataloader import RxnDataLoader
loader = RxnDataLoader(dset, batch_size=2)

from novel_arch.deep_attn.feat_type_updaters import concat_sum_atom_edge_feat, aggreg_atom_edge_no_repeat, AttnNodeEdgeAggreg, AtomEdgeReducer, bond_mean, atom_mean, bond_sum, atom_sum, A2GReducer, B2GReducer
model = DeepAtom(
        atom_aggregators=concat_sum_atom_edge_feat,
        b2g_aggregator=bond_mean(),
        a2g_aggregator=atom_mean(),
        in_feat_sizes={'atom': 12, 'bond': 7, 'global': 3},
        graph_hidden_size=64,
        graph_layers=3,
        graph_inner_layer_sizes=[[64]] * 3,
        residual=True
    )
graph, feats, feat_gens, idxs = next(iter(loader))
print(model(graph, feats, feat_gens))

exit()

# this file just makes the import system of python less painful when running scripts w/o a notebook

# import novel_arch.tests.deep_attn.bondnet_sample
from novel_arch.train_archs import deepatomsum, deepatomattn, deepatomattnnoedges, deepatomglobalattn, bondnet_original

for i in range(4):
    print('---------------- Trial {} ----------------'.format(i))
    print('------ deep atom and global attn ------')
    deepatomglobalattn()
    print('------ deep atom attn ------')
    deepatomattn()
    # print('------ deep atom attn no edges ------')
    # deepatomattnnoedges()
    # print('------ deep atom sum ------')
    # deepatomsum()
    print('------ bond net original ------')
    bondnet_original()