from novel_arch.deep_attn.data.from_csv import bdedataset_from_csv
from novel_arch.deep_attn.data.dataloader import RxnDataLoader

from novel_arch.deep_attn.model import DeepAttn
from novel_arch.deep_attn.data.initial_containers import DirectSmilesRepo, DGLwBDEMappings
from novel_arch.deep_attn.data.dataset import BDEDataset

dset = bdedataset_from_csv('/home/pmistry/Documents/research/data/ALFABET_data/acp_updated.csv', max_lines=195)

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
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error}
from train.test_on_set import TestonSet
loader = RxnDataLoader(dset, batch_size=100)
test_dset = TestonSet(loader, metric_fns, handle_mod_out=lambda x: (x * dset.val_stdev) + dset.val_mean)
# test_dset = TestonSet(loader, metric_fns)
begin_test = test_dset(model)
from torch.optim import Adam
from torch.nn import MSELoss
loss_fn = MSELoss()
op = Adam(model.parameters(), lr=0.001)

for e in range(40):
    loader = RxnDataLoader(dset, batch_size=32, shuffle=True)
    print('------------- epoch {} ------------'.format(e))
    for i, data in enumerate(loader):
        (graph, feats, feat_gens, idxs), values = data
        model.train()
        calc = model(graph, feats, feat_gens)
        # loss = loss_fn(calc, (values.view([-1, 1]) - dset.val_mean) / dset.val_stdev)
        # loss = loss_fn(calc, values.view([-1, 1]))
        loss = loss_fn((calc * dset.val_stdev) + dset.val_mean, values.view([-1, 1]))
        op.zero_grad()
        loss.backward()
        # for (n, param) in model.named_parameters():
        #     norm = param.grad.norm() if param.grad is not None else None
        #     print('aren\'t you so grad?', n, ': ', norm)
        op.step()
        print('loss lul', loss)

print('begin test result', begin_test)
print('test on set result', test_dset(model))
# loader = RxnDataLoader(dset, batch_size=1)
# (graph, feats, feat_gens, idxs), values = next(iter(loader))
# model.eval()
# print((model(graph, feats, feat_gens) * dset.val_stdev) + dset.val_mean)
# print(values)