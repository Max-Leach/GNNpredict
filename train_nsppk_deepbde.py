# %% [markdown]
# # Train DeepBDE with NSPPK-augmented features
#
# This notebook concatenates NSPPK features onto the global feature vector of the DeepBDE
# reaction graph, then trains a fresh model from scratch.
#
# **How the concatenation works:**
# - The DeepBDE reaction graph has three node types: atom, bond, and global.
# - The global node normally carries 3 scalar features (num_atoms, num_bonds, mol_weight).
# - `GlobalFeaturize` (in architecture/data/featurizers.py) already supports appending NSPPK
#   features. We just need to pass `nsppk_params` when building the dataset.
# - The resulting global feature size is `3 + 2**nbits`.
# - We tell the model about this via `in_feat_sizes={'atom': ..., 'bond': 7, 'global': 3 + 2**nbits}`.

# %% [markdown]
# ## 0. Imports

# %%
import sys, os, pickle, math, logging, csv

# Make sure imports resolve from the GNNpredict root
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.nn import MSELoss
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error,
    max_error, mean_squared_error, r2_score,
)

from architecture import construct_model
from architecture.data.dataset import BDEDataset, BDESubset
from architecture.data.dataloader import RxnDataLoader
from architecture.data.dset_generate import get_atomic_num_list
from architecture.data.featurizers import AtomFeaturize, BondFeaturize, GlobalFeaturize
from architecture.data.initial_containers import DirectSmilesRepo, DGLwBDEMappings
from architecture.item_handle import deep_bde_item_handle
from train.trainer import Trainer
from train.test.test_on_set import TestonSet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import os

file_name = "nsppk_deepbde_output.txt"

if os.path.exists(file_name):
    os.remove(file_name)
    print(f"Successfully deleted {file_name}")
else:
    print(f"{file_name} does not exist. Nothing to clear.")

# %% [markdown]
# ## 1. Configuration — edit this cell

# %%
# ── Data paths (relative to GNNpredict root) ──────────────────────────────────
TRAIN_CSV = os.path.join(ROOT, "nsppk_data", "train_subset.csv")
VAL_CSV   = os.path.join(ROOT, "nsppk_data", "val_subset.csv")

# Where to save datasets (cached to disk so you don't rebuild every run)
# and model checkpoints.
OUTPUT_DIR = os.path.join(ROOT, "nsppk_deepbde_run")

# ── NSPPK parameters ──────────────────────────────────────────────────────────
# nbits=5  →  NSPPK feature size = 2^5 = 32  (fast, matches construct_model default comment)
# nbits=13 →  NSPPK feature size = 2^13 = 8192 (from NSPPK_Example best_params; slower)
NSPPK_PARAMS = {'radius': 1, 'distance': 1, 'connector': 1, 'nbits': 5, 'sigma': 1}
NSPPK_FEAT_SIZE = 2 ** NSPPK_PARAMS['nbits']   # e.g. 32 for nbits=5

# ── Training hyperparameters ──────────────────────────────────────────────────
EPOCHS               = 100
BATCH_SIZE           = 32
LEARN_RATE           = 1e-3
REDUCELR_FACTOR      = 0.5
REDUCELR_PATIENCE    = 10
REDUCELR_THRESHOLD   = 1e-4
MIN_EPOCHS           = 30
EPOCHS_NO_MAE_STOP   = 30   # early stop if MAE hasn't improved for this many epochs
NUM_WORKERS          = 0    # set >0 only if your OS supports fork-based multiprocessing

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ── Model architecture (same defaults as the original DeepBDE) ─────────────────
GRAPH_INNER_LAYER_SIZES = [[128]*4]*5
GRAPH_HIDDEN_SIZE       = 64
FC_READOUT_SIZES        = [256] + [128]*3
ACTIVATION_FN           = None  # None = default (SiLU in model.py); or nn.ReLU, nn.ELU, etc.

# %% [markdown]
# ## 2. Dataset building
#
# `dset_generate.from_csv` hard-codes `GlobalFeaturize` without NSPPK params, so we
# replicate that function here with NSPPK support.
#
# **Dataset creation is slow** (NSPPK runs per molecule).  The code caches the finished
# datasets to `OUTPUT_DIR/dset_train/` and `OUTPUT_DIR/dset_val/` as DGL files so
# subsequent runs skip this step.

# %%
def from_csv_with_nsppk(path, nsppk_params, max_lines=None, start_line=0,
                         entry_name_to_col=None):
    """
    Like `dset_generate.from_csv` but passes `nsppk_params` to GlobalFeaturize so
    NSPPK features are concatenated to the 3 base global features.

    Column layout expected (0-indexed):
        0: serial/index   1: reactant SMILES   2: bond index
        3: frag1 SMILES   4: frag2 SMILES       5: BDE value   6: bond type
    Adjust `entry_name_to_col` if your CSV differs.
    """
    if entry_name_to_col is None:
        entry_name_to_col = {'reacs': [1], 'prods': [3, 4],
                             'broken_idx': 2, 'bde': 5, 'bondtype': 6}

    atomic_num_list = get_atomic_num_list(path, max_lines, 1,
                                          entry_name_to_col['bondtype'])

    dsr = DirectSmilesRepo()
    with open(path) as csv_file:
        csv_read = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for rxn_line in list(csv_read)[1:]:
            rsmiles = [rxn_line[i] for i in entry_name_to_col['reacs']]
            psmiles = [rxn_line[i] for i in entry_name_to_col['prods']]
            broken_bond_idxs = [int(rxn_line[entry_name_to_col['broken_idx']])]
            bde = float(rxn_line[entry_name_to_col['bde']])
            dsr.append_reaction(rsmiles, psmiles, broken_bond_idxs, bde)
            line_count += 1
            if max_lines is not None and line_count >= max_lines:
                break

    bdemap = DGLwBDEMappings(dsr)

    aprop = ['atomic_num', 'total_degree', 'total_num_hs', 'ring_of_size', 'is_in_ring']
    bprop = ['is_in_ring', 'ring_of_size', 'dative']
    gprop = ['num_atoms', 'num_bonds', 'total_weight']

    featurizers = {
        'atom':   AtomFeaturize(aprop, atomic_num_list),
        'bond':   BondFeaturize(bprop),
        'global': GlobalFeaturize(gprop, nsppk_params=nsppk_params),
    }

    dset = BDEDataset.from_initials(dsr, bdemap, featurizers)
    return dset, atomic_num_list


# ──────────────────────────────────────────────────────────────────────────────
# Build (or load from cache) train and val datasets
# ──────────────────────────────────────────────────────────────────────────────

file_name = "nsppk_deepbde_output.txt"

# The "a" mode opens the file for appending and creates it if it's missing
with open(file_name, "a") as file:
    file.write("Going to load train and val\n")

print(f"Appended to {file_name}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
train_dset_path = os.path.join(OUTPUT_DIR, "dset_train")
val_dset_path   = os.path.join(OUTPUT_DIR, "dset_val")

if os.path.exists(train_dset_path):
    print("Loading cached train dataset...")
    train_dset = BDEDataset.load(train_dset_path)
    # Recover atom_feat_size from cache metadata
    with open(os.path.join(OUTPUT_DIR, "atom_feat_size.pkl"), "rb") as f:
        ATOM_FEAT_SIZE = pickle.load(f)
else:
    print("Building train dataset (this will take a few minutes)...")
    train_dset, train_atomic_nums = from_csv_with_nsppk(TRAIN_CSV, NSPPK_PARAMS)
    # atom feature size: one-hot atomic num + 8 scalar features
    ATOM_FEAT_SIZE = len(train_atomic_nums) + 8
    train_dset.save(train_dset_path)
    with open(os.path.join(OUTPUT_DIR, "atom_feat_size.pkl"), "wb") as f:
        pickle.dump(ATOM_FEAT_SIZE, f)
    print(f"Train dataset saved to {train_dset_path}")

with open(file_name, "a") as file:
    file.write("Successfully loaded train\n")

print(f"Appended to {file_name}")

if os.path.exists(val_dset_path):
    print("Loading cached val dataset...")
    val_dset = BDEDataset.load(val_dset_path)
else:
    print("Building val dataset...")
    val_dset, _ = from_csv_with_nsppk(VAL_CSV, NSPPK_PARAMS)
    val_dset.save(val_dset_path)
    print(f"Val dataset saved to {val_dset_path}")

with open(file_name, "a") as file:
    file.write("Successfully loaded train\n")

print(f"Appended to {file_name}")

print(f"Train size: {len(train_dset)}  |  Val size: {len(val_dset)}")
print(f"Atom feat size: {ATOM_FEAT_SIZE}")
print(f"Global feat size: 3 (base) + {NSPPK_FEAT_SIZE} (NSPPK) = {3 + NSPPK_FEAT_SIZE}")


with open(file_name, "a") as file:
    file.write(f"Train size:  {len(train_dset)}  |  Val size: {len(val_dset)}\n")
    file.write(f"Atom feat size: {ATOM_FEAT_SIZE}")
    file.write(f"Global feat size: 3 (base) + {NSPPK_FEAT_SIZE} (NSPPK) = {3 + NSPPK_FEAT_SIZE}")


print(f"Appended to {file_name}")
# %% [markdown]
# ## 3. Model construction
#
# `get_std_sum_full` passes `**kwargs` down to `get_std_model`, which accepts
# `in_feat_sizes`.  We just need to set `'global': 3 + NSPPK_FEAT_SIZE`.

# %%
GLOBAL_FEAT_SIZE = 3 + NSPPK_FEAT_SIZE   # e.g. 35 for nbits=5

model = construct_model.get_std_sum_full(
    graph_inner_layer_sizes = GRAPH_INNER_LAYER_SIZES,
    graph_hidden_size       = GRAPH_HIDDEN_SIZE,
    fc_readout_sizes        = FC_READOUT_SIZES,
    activation_fn           = ACTIVATION_FN,
    in_feat_sizes           = {'atom': ATOM_FEAT_SIZE, 'bond': 7, 'global': GLOBAL_FEAT_SIZE},
)
model = model.to(DEVICE)

with open(file_name, "a") as file:
    file.write(f"Model constructed \n")

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model built. Trainable parameters: {total_params:,}")
print(model)


with open(file_name, "a") as file:
    file.write(f"Model constructed,  Trainable parameters: {total_params:,} \n")

# %% [markdown]
# ## 4. Training infrastructure

# %%
# ── BDESubset wrappers (Trainer/TestonSet expect .val_stdev / .val_mean props) ─
# Since we loaded train and val as separate full datasets, wrap each as a subset
# of itself so the downstream code gets the same interface.
train_set = BDESubset(train_dset, list(range(len(train_dset))))
valid_set = BDESubset(val_dset,   list(range(len(val_dset))))

with open(file_name, "a") as file:
    file.write(f"Subsets determined \n")

# ── Loss and metrics ─────────────────────────────────────────────────────────
loss_fn = MSELoss()

def wrap_numpy(fn):
    return lambda t1, t2: fn(t1.numpy(), t2.numpy())

metric_fns = {
    'mae':        mean_absolute_error,
    'mape':       mean_absolute_percentage_error,
    'loss':       lambda p, t: loss_fn(p, t).detach().item(),
    'max_error':  wrap_numpy(max_error),
    'mse':       wrap_numpy(mean_squared_error),
    'r2_score':   wrap_numpy(r2_score),
    'stdev_error': lambda p, t: torch.std(torch.abs(p - t)).detach().item(),
}

# Denormalize model outputs back to kcal/mol for metric computation
handle_mod_out = lambda x: (x.to(DEVICE) * train_dset.val_stdev) + train_dset.val_mean

# ── Validation tester ─────────────────────────────────────────────────────────
valid_tester = TestonSet(
    RxnDataLoader(valid_set, batch_size=100, num_workers=NUM_WORKERS),
    metric_fns,
    handle_items  = lambda items: deep_bde_item_handle(items, device=DEVICE),
    handle_mod_out = handle_mod_out,
)


with open(file_name, "a") as file:
    file.write(f"teston set \n")

# ── Callbacks ─────────────────────────────────────────────────────────────────
iter_losses = []
val_scores  = []

def iter_reporter(loss, model, epoch, i):
    iter_losses.append(loss)
    with open(os.path.join(OUTPUT_DIR, 'iter_losses'), 'wb') as f:
        pickle.dump(iter_losses, f)

def valid_reporter(valid_scores, losses, epoch, model, optim, lr_sched):
    score = valid_scores[-1]
    val_scores.append(score)
    # Always save the latest model
    torch.save(model, os.path.join(OUTPUT_DIR, 'last_model'))
    with open(os.path.join(OUTPUT_DIR, 'epoch_vals'), 'wb') as f:
        pickle.dump(val_scores, f)
    # Save every 25 epochs
    if epoch % 25 == 0:
        torch.save(model, os.path.join(OUTPUT_DIR, f'model_epoch-{epoch}'))
    # Save best model (lowest validation loss)
    best_path = os.path.join(OUTPUT_DIR, 'best_model_vals')
    try:
        with open(best_path, 'rb') as f:
            _best_epoch, best_score = pickle.load(f)
    except (FileNotFoundError, EOFError):
        best_score = {'loss': math.inf}
    if score['loss'] < best_score['loss']:
        torch.save(model, os.path.join(OUTPUT_DIR, 'best_model'))
        with open(best_path, 'wb') as f:
            pickle.dump((epoch, score), f)
        print(f"  ✓ New best at epoch {epoch}: {score}")

def should_stop(epochs_current, valid_scores):
    """Early stopping: stop if MAE hasn't improved for EPOCHS_NO_MAE_STOP epochs."""
    if epochs_current < MIN_EPOCHS or epochs_current < EPOCHS_NO_MAE_STOP:
        return False
    threshold = valid_scores[-EPOCHS_NO_MAE_STOP]['mae']
    return all(vs['mae'] >= threshold for vs in valid_scores[-EPOCHS_NO_MAE_STOP:])

# ── Optimizer and LR scheduler constructors ───────────────────────────────────
optim_construct    = lambda params: Adam(params, lr=LEARN_RATE)
lr_sched_construct = lambda o: ReduceLROnPlateau(
    o, factor=REDUCELR_FACTOR, patience=REDUCELR_PATIENCE, threshold=REDUCELR_THRESHOLD
)

# %% [markdown]
# ## 5. Train!
#
# `Trainer.__call__()` checks `OUTPUT_DIR/train_state` and resumes automatically
# if a previous run was interrupted.

# %%
trainer = Trainer(
    epochs          = EPOCHS,
    optim_construct = optim_construct,
    loss_fn         = lambda p, t: loss_fn(
                          (p.flatten() * train_set.val_stdev) + train_set.val_mean, t
                      ),
    validator       = valid_tester,
    train_loader    = RxnDataLoader(train_set, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=NUM_WORKERS),
    load_handle     = lambda items: deep_bde_item_handle(items, device=DEVICE),
    valid_reporter  = valid_reporter,
    iter_reporter   = iter_reporter,
    lr_sched_construct = lr_sched_construct,
    save_dir        = OUTPUT_DIR,
    should_stop     = should_stop,
    model           = model,
)

print(f"\nStarting training. Checkpoints → {OUTPUT_DIR}")
print(f"  epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LEARN_RATE}, device={DEVICE}\n")

with open(file_name, "a") as file:
    file.write(f"\nStarting training. Checkpoints → {OUTPUT_DIR}")
    file.write(f"  epochs={EPOCHS}, batch={BATCH_SIZE}, lr={LEARN_RATE}, device={DEVICE}\n")

trainer()

# %% [markdown]
# ## 6. Quick results summary

# %%
best_vals_path = os.path.join(OUTPUT_DIR, 'best_model_vals')
if os.path.exists(best_vals_path):
    with open(best_vals_path, 'rb') as f:
        best_epoch, best_score = pickle.load(f)
    print(f"\nBest model at epoch {best_epoch}:")
    for k, v in best_score.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
else:
    print("No best_model_vals found yet.")
