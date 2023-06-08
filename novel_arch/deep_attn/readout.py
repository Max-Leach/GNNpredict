from torch import nn
import torch
import dgl

''' set2set on ntype of nodes given a graph and features, adapted from bondnet paper '''
class Set2Set(nn.Module):
    def __init__(self, input_dim, n_iters, n_layers, ntype):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.ntype = ntype
        self.lstm = torch.nn.LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, graph, feats):
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (
                feats.new_zeros((self.n_layers, batch_size, self.input_dim)),
                feats.new_zeros((self.n_layers, batch_size, self.input_dim)),
            )

            q_s = feats.new_zeros(batch_size, self.output_dim)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_s.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feats * dgl.broadcast_nodes(graph, q, ntype=self.ntype)).sum(
                    dim=-1, keepdim=True
                )
                graph.nodes[self.ntype].data["e"] = e
                alpha = dgl.softmax_nodes(graph, "e", ntype=self.ntype)
                graph.nodes[self.ntype].data["r"] = feats * alpha
                readout = dgl.sum_nodes(graph, "r", ntype=self.ntype)
                q_s = torch.cat([q, readout], dim=-1)

            return q_s
