from novel_arch import train

# following are fns to train different architectures with the same setup

def novel(): # this is where i am playing around right now
    pass

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
        fc_activation='ReLU'
    )

    train.train_for_epochs_w_Test_MAE(model, 'original_chkpoint.pkl')