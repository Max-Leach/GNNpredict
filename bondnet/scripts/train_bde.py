# %% [markdown]
# # Train BonDNet 
# 
# In this notebook, we show how to train the BonDNet graph neural network model for bond dissociation energy (BDE) prediction. We only show how to train on CPUs. See [train_bde_distributed.py](./) for a script for training on GPUs (a single GPU or distributed training on multiple GPUs). 

# %%
import torch
from torch.nn import MSELoss
from bondnet.data.dataset import ReactionNetworkDataset
from bondnet.data.dataloader import DataLoaderReactionNetwork
from bondnet.data.featurizer import AtomFeaturizerMinimum, BondAsNodeFeaturizerMinimum, GlobalFeaturizer
from bondnet.data.grapher import HeteroMoleculeGraph
from bondnet.data.dataset import train_validation_test_split
from bondnet.model.gated_reaction_network import GatedGCNReactionNetwork
from bondnet.scripts.create_label_file import read_input_files
from bondnet.model.metric import WeightedL1Loss
from bondnet.utils import seed_torch

# %% [markdown]
# ## Dataset 
# 
# We work with a small dataset consisting of 200 BDEs for netural and charged molecules. The dataset is specified in three files:
# - `molecules.sdf` This file contains all the molecules (both reactants and products) in the bond dissociation reactions. The molecules are specified in SDF format. 
# - `molecule_attributes.yaml` This file contains extra molecular attributes (charges here) for molecules given in `molecules.sdf`. Some molecular attributes can be inferred from its SDF block, and they are overrode by the attributes specified in the `molecule_attributes.yaml` file.  
# - `reactions.csv` This file list the bond dissociation reations formed by the molecules given in `molecules.sdf`. Each line lists the reactant, products, and BDE of a reaction. The reactant and products are specified by their index in `molecules.sdf`. 
# 
# See [here](./examples/train) for the three files used in this notebook. 

# %% [markdown]
# #### Grapher 
# 
# BondNet is graph neutral network model that takes atom features (e.g. atom type), bond features (e.g. whether a bond is in a ring), and global features (e.g. total charge) as input. We extract the features for a molecule using a grapher.

# %%
def get_grapher():
    atom_featurizer = AtomFeaturizerMinimum()
    bond_featurizer = BondAsNodeFeaturizerMinimum()
    
    # our example dataset contains molecules of charges -1, 0, and 1
    global_featurizer = GlobalFeaturizer(allowed_charges=[-1, 0, 1])

    grapher = HeteroMoleculeGraph(atom_featurizer, bond_featurizer, global_featurizer)
    
    return grapher

# %% [markdown]
# #### Read dataset 
# 
# Let's now read the dataset and featurize the molecules using the above defined grapher. The dataset is split into a training set (80%), validation set (10%), and test set (10%). We will train our model using the training set, stop the training using the validation set, and report error on the test set. 

# %%
# seed random number generators 
seed_torch()

mols, attrs, labels = read_input_files(
    'bondnet/scripts/examples/train/molecules.sdf', 
    'bondnet/scripts/examples/train/molecule_attributes.yaml', 
    'bondnet/scripts/examples/train/reactions.yaml', 
)
dataset = ReactionNetworkDataset(
    grapher=get_grapher(),
    molecules=mols,
    labels=labels,
    extra_features=attrs
)

trainset, valset, testset = train_validation_test_split(dataset, validation=0.1, test=0.1)

# we train with a batch size of 100
train_loader = DataLoaderReactionNetwork(trainset, batch_size=100,shuffle=True)
val_loader = DataLoaderReactionNetwork(valset, batch_size=len(valset), shuffle=False)
test_loader = DataLoaderReactionNetwork(testset, batch_size=len(testset), shuffle=False)

# %% [markdown]
# ## Model 
# 
# We create the BonDNet model by instantiating the `GatedGCNReactionNetwork` class and providing the parameters defining the model structure. 
# - `embedding_size` The size to unify the atom, bond, and global feature length.
# - `gated_num_layers` Number of graph to graph module to learn molecular representation. 
# - `gated_hidden_size` Hidden layer size in the graph to graph modules. 
# - `gated_activation` Activation function appleid after the hidden layers in the graph to graph modules. 
# - `fc_num_layers` Number of hidden layers of the fully connected network to map reaction feature to the BDE. The reaction feature is obtained as the differece of the features between the products and the reactant. 
# - `fc_hidden_size` Size of the hidden layers. 
# - `fc_activation` Activation function applied after the hidden layers. 
# 
# There are other arguments (e.g. residual connection, dropout ratio, batch norm) that can be specified to fine control the model. See the documentation of the `GatedGCNReactionNetwork` for more information.  

# %%
model = GatedGCNReactionNetwork(
    in_feats=dataset.feature_size,
    embedding_size=24,
    gated_num_layers=3,
    gated_hidden_size=[64, 64, 64],
    gated_activation="ReLU",
    fc_num_layers=2,
    fc_hidden_size=[128, 64],
    fc_activation='ReLU'
)

# %% [markdown]
# ## Train the model 
# 
# Before going to the main training loop, we define two functions: `train` and `evaluate` that will be used later. 
# 
# The `train` function optimizes the model parameters for an epoch. We note that our target BDEs are centered and then normalized by the standard deviation (done in the `ReactionNetworkDataset`.) So to measure the mean absolute error, we need to multiply the standard deviation back. This is acheived achieved by the `WeightedL1Loss` function passed as `metric_fn`.   

# %%
def train(optimizer, model, nodes, data_loader, loss_fn, metric_fn):

    model.train()

    epoch_loss = 0.0
    accuracy = 0.0
    count = 0.0

    for it, (batched_graph, label) in enumerate(data_loader):
        feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
        target = label["value"]
        stdev = label["scaler_stdev"]

        pred = model(batched_graph, feats, label["reaction"])
        pred = pred.view(-1)

        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.detach().item()
        accuracy += metric_fn(pred, target, stdev).detach().item()
        count += len(target)
    
    epoch_loss /= it + 1
    accuracy /= count

    return epoch_loss, accuracy

# %% [markdown]
# The `evaluate` function computes the mean absolute error for the validation set (or test set).

# %%
def evaluate(model, nodes, data_loader, metric_fn):
    model.eval()

    with torch.no_grad():
        accuracy = 0.0
        count = 0.0

        for batched_graph, label in data_loader:
            feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
            target = label["value"]
            stdev = label["scaler_stdev"]

            pred = model(batched_graph, feats, label["reaction"])
            pred = pred.view(-1)

            accuracy += metric_fn(pred, target, stdev).detach().item()
            count += len(target)

    return accuracy / count

# %% [markdown]
# Now, we have all the ingredients to train the model. 
# 
# We optimize the model parameters by minimizing a mean squared error loss function using the `Adam` optimizer with a learning rate of `0.001`. Here we train the model for 20 epochs; save the best performing model that gets the smallest mean absolute error on the validation set; and finally test model performance on the test set. 

# %%
# optimizer, loss function and metric function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = MSELoss(reduction="mean")
metric = WeightedL1Loss(reduction="sum")

feature_names = ["atom", "bond", "global"]
best = 1e10
num_epochs = 20

# main training loop
print("# Epoch     Loss         TrainAcc        ValAcc")
for epoch in range(num_epochs):

    # train on training set 
    loss, train_acc = train( optimizer, model, feature_names, train_loader, loss_func, metric)

    # evaluate on validation set
    val_acc = evaluate(model, feature_names, val_loader, metric)

    # save checkpoint for best performing model 
    is_best = val_acc < best
    if is_best:
        best = val_acc
        torch.save(model.state_dict(), 'checkpoint.pkl')
        
    print("{:5d}   {:12.6e}   {:12.6e}   {:12.6e}".format(epoch, loss, train_acc, val_acc))


# load best performing model and test it's performance on the test set
checkpoint = torch.load("checkpoint.pkl")
model.load_state_dict(checkpoint)
test_acc = evaluate(model, feature_names, test_loader, metric)

print("TestAcc: {:12.6e}".format(test_acc))

# %%



