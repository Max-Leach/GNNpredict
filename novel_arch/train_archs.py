from novel_arch import train

# following are fns to train different architectures with the same setup

from novel_arch.deep_attn.model import DeepAtomSum

def deepatomsum():
    model = DeepAtomSum(
        in_feat_sizes=train.dataset.feature_size, 
        # embedding_size=64,
        graph_hidden_size=64,
        graph_layers=3,
        graph_inner_layer_sizes=[[64]] * 3,
        residual=True
    )

    train.train_for_epochs_w_Test_MAE(model, 'deepatomsum.pkl', lr=0.001)

from novel_arch.dmpnn_like.model import DMPNNLike
from novel_arch.dmpnn_like.directed_conv import DMPNNPropag

def dmpnn_like():
    model = DMPNNLike(
        in_feats=train.dataset.feature_size,
        dbond_feat_size=64, # atom, bond -> dbond features
        node_types=["atom", "bond"], #, "global"],
        gated_residual=False, # this appears to improve model training performance
        set2set_ntypes_direct=None, # remove global in general

        embedding_size=64,
        gated_num_layers=3,
        gated_hidden_size=[64, 64, 64],
        gated_activation="ReLU",
        fc_num_layers=2,
        fc_hidden_size=[128, 64],
        fc_activation='ReLU',
        conv_op=DMPNNPropag
    )

    train.train_for_epochs_w_Test_MAE(model, 'dmpnn_like_chkpoint.pkl', lr=0.0015)

from novel_arch.archic_0.model import GatedGCNReactionNetworkDMPNN
from novel_arch.archic_0.directed_conv import GatedGCNConvDMPNN

def archic_0():
    gated_hidden_size = [64, 64, 64,]
    model = GatedGCNReactionNetworkDMPNN(
        in_feats=train.dataset.feature_size,
        dbond_feat_size=64, # atom, bond -> dbond features
        node_types=["atom", "bond", "global"],
        gated_residual=True, # this appears to improve model training performance

        embedding_size=24,
        gated_num_layers=len(gated_hidden_size),
        gated_hidden_size=gated_hidden_size,
        gated_activation="ReLU",
        fc_num_layers=2,
        fc_hidden_size=[128, 64],
        fc_activation='ReLU',
        conv_op=GatedGCNConvDMPNN
    )

    train.train_for_epochs_w_Test_MAE(model, 'archic-0_chkpoint.pkl', lr=0.0015, num_epochs=30)

from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
def bondnet_original():
    model = GatedGCNReactionNetwork(
        in_feats=train.dataset.feature_size,
        embedding_size=24,
        gated_num_layers=3,
        gated_hidden_size=[64, 64, 64],
        gated_activation="ReLU",
        fc_num_layers=2,
        fc_hidden_size=[128, 64],
        fc_activation='ReLU',
        fc_batch_norm=True,
    )

    train.train_for_epochs_w_Test_MAE(model, 'original_chkpoint.pkl')