## adapted from bondnet/scripts/train_bde.ipynb

from bondnet.data.featurizer import AtomFeaturizerMinimum, BondAsNodeFeaturizerMinimum, GlobalFeaturizer
from bondnet.data.grapher import HeteroMoleculeGraph

from bondnet.data.dataset import ReactionNetworkDataset
from bondnet.data.dataloader import DataLoaderReactionNetwork

from bondnet.scripts.create_label_file import read_input_files
from bondnet.data.dataset import train_validation_test_split
from bondnet.utils import seed_torch

from bondnet.model.metric import WeightedL1Loss

from torch.nn import MSELoss
import torch

def get_grapher():
    atom_featurizer = AtomFeaturizerMinimum()
    bond_featurizer = BondAsNodeFeaturizerMinimum()
    
    # our example dataset contains molecules of charges -1, 0, and 1
    global_featurizer = GlobalFeaturizer(allowed_charges=[-1, 0, 1])

    grapher = HeteroMoleculeGraph(atom_featurizer, bond_featurizer, global_featurizer)
    
    return grapher

# constant seed setup
seed_torch()

## dataset loading
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
val_loader = DataLoaderReactionNetwork(valset, batch_size=len(valset), shuffle=False) # batch size is the entire set likely because they are so small in this case
test_loader = DataLoaderReactionNetwork(testset, batch_size=len(testset), shuffle=False)

import dgl

# The `train` function optimizes the model parameters for an epoch. We note that our target BDEs are centered and then normalized by the standard deviation (done in the `ReactionNetworkDataset`.) So to measure the mean absolute error, we need to multiply the standard deviation back. This is acheived achieved by the `WeightedL1Loss` function passed as `metric_fn`.
def train(optimizer, model, nodes, data_loader, loss_fn, metric_fn):

    model.train()

    epoch_loss = 0.0
    accuracy = 0.0
    count = 0.0

    # normal training
    for it, (batched_graph, label) in enumerate(data_loader):
        feats = {nt: batched_graph.nodes[nt].data["feat"] for nt in nodes}
        # print(feats['atom'].shape, feats['bond'].shape, feats['global'].shape) # here!
        target = label["value"]
        stdev = label["scaler_stdev"]

        pred = model(batched_graph, feats, label["reaction"])
        pred = pred.view(-1)

        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        # grads = {}
        # for (n, param) in model.named_parameters():
        #     norm = param.grad.norm() if param.grad is not None else None
        #     print('aren\'t you so grad?', n, ': ', norm)
            # if norm is not None:
            #     continue
            # print(n, ':', norm)
        # print('grad magnitudes', grads)
        optimizer.step()

        epoch_loss += loss.detach().item()
        accuracy += metric_fn(pred, target, stdev).detach().item()
        count += len(target)
    
    # single point train test
    # rxn, reac_id, vals = trainset[0]
    # sing_rxn, graffs = rxn.subselect_reactions([reac_id])
    # graff = dgl.batch(graffs)
    # feats = {nt: graff.nodes[nt].data["feat"] for nt in nodes}
    # ener = vals['value'] * vals['scaler_stdev'] + vals['scaler_mean']
    # target = ener

    # pred = model(graff, feats, sing_rxn)
    # pred = pred.view(-1)

    # loss = loss_fn(pred, target)
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()

    # epoch_loss += loss.detach().item()
    # accuracy += metric_fn(pred, target, vals['scaler_stdev']).detach().item()
    # count += 1
    # it = 0

    # ---------------------
    
    epoch_loss /= it + 1
    accuracy /= count

    return epoch_loss, accuracy

# The `evaluate` function computes the mean absolute error for the given set (typically validation or test)
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

# shove in a compatibleGNN, train for multiple epochs to get best model saved, return test set accuracy
def train_for_epochs_w_Test_MAE(model, checkpoint_name, lr=0.001, num_epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = MSELoss(reduction="mean")
    metric = WeightedL1Loss(reduction="sum")

    feature_names = ["atom", "bond", "global"]
    best = 1e10

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
            torch.save(model.state_dict(), checkpoint_name)
            
        print("{:5d}   {:12.6e}   {:12.6e}   {:12.6e}".format(epoch, loss, train_acc, val_acc))
    
    # load best performing model (on valset) and test it's performance on the test set
    checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint)
    test_acc = evaluate(model, feature_names, test_loader, metric)

    print("TestAcc: {:12.6e}".format(test_acc))
    return test_acc