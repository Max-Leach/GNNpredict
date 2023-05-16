## adapted from BonDnet paper

from bondnet.layer.utils import LinearN
from torch import nn
import torch
import dgl
from typing import Callable, Union, Dict

class GatedGCNConvDMPNN(nn.Module):
    """
    Gated GCN layer.

    Update directed bond edges (2 per bond), then update global state

    Args:
        input_dim: input feature dimension
        output_dim: output feature dimension
        num_fc_layers: number of NN layers to transform input to output. In `Residual
            Gated Graph ConvNets` the number of layers is set to 1. Here we make it a
            variable to accept any number of layers.
        graph_norm: whether to apply the graph norm proposed in
            Benchmarking Graph Neural Networks (https://arxiv.org/abs/2003.00982)
        batch_norm: whether to apply batch normalization
        activation: activation function
        residual: whether to add residual connection as in the ResNet:
            Deep Residual Learning for Image Recognition (https://arxiv.org/abs/1512.03385)
        dropout: dropout ratio. Note, dropout is applied after residual connection.
            If `None`, do not apply dropout.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_fc_layers: int = 1,
        graph_norm: bool = False,
        batch_norm: bool = True,
        activation: Callable = nn.ReLU(),
        residual: bool = False,
        dropout: Union[float, None] = None,
    ):
        super().__init__()
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.activation = activation
        self.residual = residual

        if input_dim != output_dim:
            self.residual = False

        out_sizes = [output_dim] * num_fc_layers
        acts = [activation] * (num_fc_layers - 1) + [nn.Identity()]
        use_bias = [True] * num_fc_layers

        # A, B, ... I are equivalent of phi1, etc in bondnet
        # but we only do d_bond and global hidden state evolution
        # self.A = LinearN(input_dim, out_sizes, acts, use_bias)
        self.B = LinearN(input_dim, out_sizes, acts, use_bias)
        self.C = LinearN(input_dim, out_sizes, acts, use_bias)
        # self.D = LinearN(input_dim, out_sizes, acts, use_bias)
        # self.E = LinearN(input_dim, out_sizes, acts, use_bias)
        self.F = LinearN(input_dim, out_sizes, acts, use_bias)
        # self.G = LinearN(output_dim, out_sizes, acts, use_bias)
        self.H = LinearN(output_dim, out_sizes, acts, use_bias)
        self.I = LinearN(input_dim, out_sizes, acts, use_bias)

        if self.batch_norm:
            self.bn_node_h = nn.BatchNorm1d(output_dim)
            self.bn_node_e = nn.BatchNorm1d(output_dim)
            self.bn_node_u = nn.BatchNorm1d(output_dim)

        delta = 1e-3
        if dropout is None or dropout < delta:
            self.dropout = nn.Identity()
        else:
            self.dropout = nn.Dropout(dropout)

    # @staticmethod
    # def reduce_fn_a2b(nodes):
    #     """
    #     Reduce `Eh_j` from atom nodes to bond nodes.

    #     Expand dim 1 such that every bond has two atoms connecting to it.
    #     This is to deal with the special case of single atom graph (e.g. H+).
    #     For such graph, an artificial bond is created and connected to the atom in
    #     `grapher`. Here, we expand it to let each bond connecting to two atoms.
    #     This is necessary because, otherwise, the reduce_fn wil not work since
    #     dimension mismatch.
    #     """
    #     x = nodes.mailbox["Eh_j"]
    #     if x.shape[1] == 1:
    #         x = x.repeat_interleave(2, dim=1)

    #     return {"Eh_j": x}

    # @staticmethod
    # def message_fn(edges):
    #     return {"Eh_j": edges.src["Eh_j"], "e": edges.src["e"]}

    # @staticmethod
    # def reduce_fn(nodes):
    #     Eh_i = nodes.data["Eh"]
    #     e = nodes.mailbox["e"]
    #     Eh_j = nodes.mailbox["Eh_j"]

    #     # TODO select_not_equal is time consuming; it might be improved by passing node
    #     #  index along with Eh_j and compare the node index to select the different one
    #     Eh_j = select_not_equal(Eh_j, Eh_i)
    #     sigma_ij = torch.sigmoid(e)  # sigma_ij = sigmoid(e_ij)

    #     # (sum_j eta_ij * Ehj)/(sum_j' eta_ij') <= dense attention
    #     h = torch.sum(sigma_ij * Eh_j, dim=1) / (torch.sum(sigma_ij, dim=1) + 1e-6)

    #     return {"h": h}

    def forward(
        self,
        g: dgl.DGLGraph,
        feats: Dict[str, torch.Tensor],
        norm_atom: torch.Tensor = None,
        norm_bond: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            g: the graph
            feats: node features. Allowed node types are `atom`, `bond` and `global`.
            norm_atom: values used to normalize atom features as proposed in graph norm.
            norm_bond: values used to normalize bond features as proposed in graph norm.

        Returns:
            updated node features.
        """

        g = g.local_var()

        # h = feats["atom"]
        e = feats["d_bond"]
        u = feats["global"]

        # for residual connection
        # h_in = h
        e_in = e
        u_in = u

        # g.nodes["atom"].data.update({"Ah": self.A(h), "Dh": self.D(h), "Eh": self.E(h)})
        g.nodes["d_bond"].data.update({"Be": self.B(e)})
        g.nodes["global"].data.update({"Cu": self.C(u), "Fu": self.F(u)})

        # update bond feature e
        g.multi_update_all(
            {
                # "a2b": (fn.copy_u("Ah", "m"), fn.sum("m", "e")),  # A * (h_i + h_j)
                #"db2db" : (blah) # we will likely need to add a propagation to other bonds here
                "b2b": (fn.copy_u("Be", "m"), fn.sum("m", "e")),  # B * e_ij # self bond update
                "g2b": (fn.copy_u("Cu", "m"), fn.sum("m", "e")),  # C * u
            },
            "sum",
        )

        e = g.nodes["d_bond"].data["e"]
        if self.graph_norm:
            e = e * norm_bond
        if self.batch_norm:
            e = self.bn_node_e(e)
        e = self.activation(e)
        if self.residual:
            e = e_in + e
        g.nodes["d_bond"].data["e"] = e

        # update atom feature h

        # Copy Eh to bond nodes, without reduction.
        # This is the first arrow in: Eh_j -> bond node -> atom i node
        # The second arrow is done in self.message_fn and self.reduce_fn below
        # g.update_all(fn.copy_u("Eh", "Eh_j"), self.reduce_fn_a2b, etype="a2b")

        # g.multi_update_all(
        #     {
        #         "a2a": (fn.copy_u("Dh", "m"), fn.sum("m", "h")),  # D * h_i
        #         "b2a": (self.message_fn, self.reduce_fn),  # e_ij [Had] (E * hj)
        #         "g2a": (fn.copy_u("Fu", "m"), fn.sum("m", "h")),  # F * u
        #     },
        #     "sum",
        # )

        # h = g.nodes["atom"].data["h"]
        # if self.graph_norm:
        #     h = h * norm_atom
        # if self.batch_norm:
        #     h = self.bn_node_h(h)
        # h = self.activation(h)
        # if self.residual:
        #     h = h_in + h
        # g.nodes["atom"].data["h"] = h

        # update global feature u
        # g.nodes["atom"].data.update({"Gh": self.G(h)})
        g.nodes["d_bond"].data.update({"He": self.H(e)})
        g.nodes["global"].data.update({"Iu": self.I(u)})
        g.multi_update_all(
            {
                # "a2g": (fn.copy_u("Gh", "m"), fn.mean("m", "u")),  # G * (mean_i h_i)
                "db2g": (fn.copy_u("He", "m"), fn.mean("m", "u")),  # H * (mean_ij e_ij)
                "g2g": (fn.copy_u("Iu", "m"), fn.sum("m", "u")),  # I * u
            },
            "sum",
        )
        u = g.nodes["global"].data["u"]
        # do not apply batch norm if it there is only one graph
        if self.batch_norm and u.shape[0] > 1:
            u = self.bn_node_u(u)
        u = self.activation(u)
        if self.residual:
            u = u_in + u

        # dropout
        # h = self.dropout(h)
        e = self.dropout(e)
        u = self.dropout(u)

        # feats = {"atom": h, "bond": e, "global": u}
        feats = {"d_bond": e, "global": u}

        return feats