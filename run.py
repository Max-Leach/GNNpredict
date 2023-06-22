from novel_arch.deep_attn.data.from_csv import bdedataset_from_csv
from novel_arch.deep_attn.model import DeepAttn
from novel_arch.deep_attn.data.dataloader import DataLoader
from novel_arch.deep_attn.data.initial_containers import DirectSmilesRepo, DGLwBDEMappings
from novel_arch.deep_attn.data.dataset import BDEDataset

dset = bdedataset_from_csv('/home/pmistry/Documents/research/data/ALFABET_data/acp_updated.csv', max_lines=200)
from novel_arch.deep_attn.data.dataloader import RxnDataLoader

from novel_arch.deep_attn.feat_type_updaters import concat_sum_atom_edge_feat, aggreg_atom_edge_no_repeat, AttnNodeEdgeAggreg, AtomEdgeReducer, bond_mean, atom_mean, bond_sum, atom_sum, A2GReducer, B2GReducer
model = DeepAttn(
        atom_aggregators=concat_sum_atom_edge_feat,
        b2g_aggregator=bond_mean(),
        a2g_aggregator=atom_mean(),
        in_feat_sizes={'atom': 12, 'bond': 7, 'global': 3},
        graph_hidden_size=64,
        graph_layers=3,
        graph_inner_layer_sizes=[[64]] * 3,
        residual=True
    )
from torch.optim import Adam
from torch.nn import MSELoss
loss_fn = MSELoss()
op = Adam(model.parameters(), lr=0.001)
print(dset.val_stdev, dset.val_mean)

for e in range(20):
    loader = RxnDataLoader(dset, batch_size=64)
    print('------------- epoch {} ------------'.format(e))
    for i, data in enumerate(loader):
        (graph, feats, feat_gens, idxs), values = data
        calc = model(graph, feats, feat_gens)
        loss = loss_fn(calc, (values.view([-1, 1]) - dset.val_mean) / dset.val_stdev)
        op.zero_grad()
        loss.backward()
        # for (n, param) in model.named_parameters():z
        #     norm = param.grad.norm() if param.grad is not None else None
        #     print('aren\'t you so grad?', n, ': ', norm)
        op.step()
        print('loss lul', loss)

loader = RxnDataLoader(dset, batch_size=1)
(graph, feats, feat_gens, idxs), values = next(iter(loader))
# print('fake', graph, feats, feat_gens, idxs, values)
model.eval()
print((model(graph, feats, feat_gens) * dset.val_stdev) + dset.val_mean)
print(values)
exit()

# dsr = DirectSmilesRepo()
# dsr.append_reaction(['NCCCC(=O)O'], ['[CH2]CCC(=O)O', '[NH2]'], [0], 0.2)
# dsr.append_reaction(['NCCCC(=O)O'], ['[CH2]CC(=O)O', '[CH2]N'], [1], 0.4)

# dsr.append_reaction(['CN(N)CCCC(C)(C)C'], ['[CH2]CC(C)(C)C', '[CH2]N(C)N'], [3], 0.3)
# dsr.append_reaction(['COC(=O)OCCC#N'], ['CO[C]=O', 'N#CCC[O]'], [3], -0.2)
# dsr.append_reaction(['C/C=C/[C@@H](C)CCC#N'], ['[H]', 'C/C=C/[C@@H](C)[CH]CC#N'], [17], 0.2)

# bdemap = DGLwBDEMappings(dsr)
# from novel_arch.deep_attn.data.featurizers import AtomFeaturize, BondFeaturize, GlobalFeaturize
# aprop = ['atomic_num', 'total_degree', 'total_num_hs', 'ring_of_size', 'is_in_ring']
# bprop = ['is_in_ring', 'ring_of_size', 'dative']
# gprop = ['num_atoms', 'num_bonds', 'total_weight']
# dset = BDEDataset(dsr, bdemap, featurizers={'atom' : AtomFeaturize(aprop, [1, 6, 7, 8]), 'bond' : BondFeaturize(bprop), 'global' : GlobalFeaturize(gprop),}, load_graphs=False)

# this file just makes the import system of python less painful when running scripts w/o a notebook

# import novel_arch.tests.deep_attn.bondnet_sample
from novel_arch.train_archs import deepattnsum, deepattnattn, deepattnattnnoedges, deepattnglobalattn, bondnet_original

for i in range(4):
    print('---------------- Trial {} ----------------'.format(i))
    print('------ deep attn and global attn ------')
    deepattnglobalattn()
    print('------ deep attn attn ------')
    deepattnattn()
    # print('------ deep attn attn no edges ------')
    # deepattnattnnoedges()
    # print('------ deep attn sum ------')
    # deepattnsum()
    print('------ bond net original ------')
    bondnet_original()