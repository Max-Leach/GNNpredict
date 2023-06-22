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
from train.eval_metrics import eval_metrics_over_loader
loader = RxnDataLoader(dset, batch_size=24)
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
metric_fns = {'mae': mean_absolute_error, 'mape': mean_absolute_percentage_error}
print(eval_metrics_over_loader(model, loader, metric_fns, handle_mod_out=lambda x: (x * dset.val_stdev) + dset.val_mean))
from torch.optim import Adam
from torch.nn import MSELoss
loss_fn = MSELoss()
op = Adam(model.parameters(), lr=0.001)
# print(dset.val_stdev, dset.val_mean)

for e in range(20):
    loader = RxnDataLoader(dset, batch_size=24)
    print('------------- epoch {} ------------'.format(e))
    for i, data in enumerate(loader):
        (graph, feats, feat_gens, idxs), values = data
        model.train()
        calc = model(graph, feats, feat_gens)
        loss = loss_fn(calc, (values.view([-1, 1]) - dset.val_mean) / dset.val_stdev)
        op.zero_grad()
        loss.backward()
        # for (n, param) in model.named_parameters():
        #     norm = param.grad.norm() if param.grad is not None else None
        #     print('aren\'t you so grad?', n, ': ', norm)
        op.step()
        print('loss lul', loss)

loader = RxnDataLoader(dset, batch_size=24)
print(eval_metrics_over_loader(model, loader, metric_fns, handle_mod_out=lambda x: (x * dset.val_stdev) + dset.val_mean))
loader = RxnDataLoader(dset, batch_size=1)
(graph, feats, feat_gens, idxs), values = next(iter(loader))
model.eval()
print((model(graph, feats, feat_gens) * dset.val_stdev) + dset.val_mean)
print(values)