import pandas as pd
import networkx as nx
import numpy as np
import pickle
import time
from itertools import product
from scipy.sparse import vstack, csr_matrix

from rdkit import Chem
from nsppk import NodeNSPPK

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel

df_train = pd.read_csv("data/train_subset.csv")
df_test = pd.read_csv("data/test_subset.csv")

def smiles_to_graph(cann):
    mol = Chem.AddHs(Chem.MolFromSmiles(cann))
    G = nx.Graph()

    for atom in mol.GetAtoms():
        atom_features = {
            'label': atom.GetSymbol(),
            'vec': np.array([
                float(atom.GetAtomicNum()),
                float(atom.GetFormalCharge()),
                float(atom.GetMass()),
                float(atom.GetHybridization()),
                float(atom.GetNumExplicitHs()),
                float(atom.GetDegree()),
                float(atom.GetIsAromatic()),
                float(atom.IsInRing())
            ])
        }
        G.add_node(atom.GetIdx(), **atom_features)

    for bond in mol.GetBonds():
        bond_features = {'label': str(bond.GetBondType())}
        G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), **bond_features)

    return G

def get_bond_atoms(product_smiles, bond_index):
    product_mol = Chem.AddHs(Chem.MolFromSmiles(product_smiles))
    bond = product_mol.GetBondWithIdx(bond_index)
    return bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

def extract_features(X, data, radius, distance, connector, nbits, sigma):
    vectorizer = NodeNSPPK(radius=radius, distance=distance, connector=connector, nbits=nbits, dense=False, sigma=sigma)
    X_transformed = vectorizer.fit_transform(X)

    product_sums = []
    for index in range(len(X)):
        product_smiles = data['Parentsmiles'].iloc[index]
        bond_index = int(data['BondIndex'].iloc[index])
        atom_indices = get_bond_atoms(product_smiles, bond_index)

        product_row_1 = X_transformed[index][atom_indices[0]]
        product_row_2 = X_transformed[index][atom_indices[1]]
        product_sums.append(product_row_1 + product_row_2)

    X_sparse = vstack([csr_matrix(x.reshape(1, -1)) if x.ndim == 1 else csr_matrix(x) for x in product_sums])
    return X_sparse

sample_sizes = [10, 50, 100, 250]
samples_train = {size: df_train.sample(n=min(int(size), len(df_train)), random_state=42) for size in sample_sizes}

df_test = df_test[:1000]

X_test = df_test['Parentsmiles'].apply(smiles_to_graph)
y_test = df_test['BDH']


for n in sample_sizes:
    subset_train = samples_train[n]

    X_train = subset_train['Parentsmiles'].apply(smiles_to_graph)
    y_train = subset_train['BDH']

    best_params = {'radius': 1, 'distance': 1, 'connector': 1, 'nbits': 13, 'sigma': 1}

    t1 = time.time()
    X_train_transformed = extract_features(X_train, subset_train, **best_params)

    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('regressor', XGBRegressor(objective='reg:absoluteerror'))
    ])

    X_test_transformed = extract_features(X_test, df_test, **best_params)

    pipeline.fit(X_train_transformed, y_train)

    y_pred = pipeline.predict(X_test_transformed)

    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = root_mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

#    with open(f"model_{n}_samples.pkl", "wb") as f:
#        pickle.dump(pipeline, f)

    t2 = time.time()
    print(f"Samples: {n},  MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, Time: {t2-t1:.2f}s\n")


###
# Hyperparameter tuning

xgb_param_grid = {
    'regressor__n_estimators': [100, 500, 1000],
    'regressor__max_depth': [100],
    'regressor__learning_rate': [0.01],
    'regressor__reg_alpha': [0, 0.01, 0.1],  # L1 regularization
    'regressor__reg_lambda': [1, 1.5, 2]     # L2 regularization
}

for n in sample_sizes:
    print("N:",n)
    subset_train = samples_train[n]

    X_train = subset_train['Parentsmiles'].apply(smiles_to_graph)
    y_train = subset_train['BDH']

    best_params = {'radius': 1, 'distance': 1, 'connector': 1, 'nbits': 13, 'sigma': 1}

    t1 = time.time()
    X_train_transformed = extract_features(X_train, subset_train, **best_params)

    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('regressor', XGBRegressor(objective='reg:absoluteerror'))
    ])

    X_test_transformed = extract_features(X_test, df_test, **best_params)

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=xgb_param_grid,
        scoring='neg_mean_absolute_error',  # minimize MAE
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train_transformed, y_train)
    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test_transformed)

    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = root_mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    t2 = time.time()
    print(f"Samples: {n}, Best XGB params: {grid_search.best_params_}")
    print(f"MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, Time: {t2-t1:.2f}s\n")

param_grid = {
    'radius': [0, 1, 2],
    'distance': [1, 2, 3],
    'connector': [1, 2],
    'nbits': [13]
}

best_score = float("inf")
best_params = None

subset_train = samples_train[250]

all_combinations = list(product(*param_grid.values()))
print("Total number of fits: ", len(all_combinations))

for params in all_combinations:
    r, d, c, n = params
    print(f"radius: {r}, distance: {d}, connector: {c}, nbits: {n}")

    X_train = subset_train['Parentsmiles'].apply(smiles_to_graph)
    y_train = subset_train['BDH']

    t1 = time.time()
    X_train_transformed = extract_features(X_train, subset_train, r, d, c, n, r)

    model = XGBRegressor(objective='reg:absoluteerror')
    
    pipeline = Pipeline([
        ('scaler', StandardScaler(with_mean=False)),
        ('regressor', model)
    ])

    X_test_transformed = extract_features(X_test, df_test, r, d, c, n, r)


    pipeline.fit(X_train_transformed, y_train)

    y_pred = pipeline.predict(X_test_transformed)

    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = root_mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    if test_mae < best_score:
        best_score = test_mae
        best_params = {'radius': r, 'distance': d, 'connector': c, 'nbits': n}

    t2 = time.time()
    print(f"MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}, Time: {t2-t1:.2f}s\n")

print("Best score: ", best_score)
print("Best params: ", best_params)


# Extract features
subset_train = samples_train[250]
len(subset_train)

X_train = subset_train['Parentsmiles'].apply(smiles_to_graph)
y_train = subset_train['BDH']

params = {'radius': 1, 'distance': 1, 'connector': 1, 'nbits': 5, 'sigma': 1}

t1 = time.time()
X_train_transformed = extract_features(X_train, subset_train, **params)
t2 = time.time()

print("Time taken: ", t2-t1, "seconds")

t1 = time.time()
X_test_transformed = extract_features(X_test, df_test, **params)
t2 = time.time()

print("Time taken: ", t2-t1, "seconds")


# The following is what I am trying to work on
# It is trying to combine my features with the graph features from my Graph Isomorphism Network (GNN)
# I need help combining the features:

import scipy.sparse as sp
sp.issparse(X_train_transformed)

import torch
import scipy.sparse as sp

def get_rxn_features_with_nsppk(batched_graph, batched_feats, rxns, my_feats_matrix):
    """
    Integrates per-graph NSPPK features (sparse or dense) into reaction features.
    
    Parameters:
        batched_graph : DGLGraph
            Batched graph containing all molecules.
        batched_feats : dict
            Original node features (keys: 'atom', 'bond', 'global').
        rxns : list
            List of RxnFeatGenerator objects (e.g., BondDissociate).
        my_feats_matrix : scipy.sparse.csr_matrix or np.ndarray
            NSPDK features summed per graph (shape: [N_graphs, feature_dim])
    
    Returns:
        rxn_features : list of dicts
            Reaction features with NSPPK features concatenated as 'global'.
    """
    if sp.issparse(my_feats_matrix):
        my_dense_feats = torch.tensor(my_feats_matrix.todense(), dtype=torch.float32)
    else:
        my_dense_feats = torch.tensor(my_feats_matrix, dtype=torch.float32)
    
    # Attach original features to the batched graph
    for nt in batched_feats:
        batched_graph.nodes[nt].data.update({'ft': batched_feats[nt]})
    
    # Unbatch the graph
    graphs = dgl.unbatch(batched_graph)
    
    # Prepare per-graph features including NSPPK as 'global'
    my_feats = []
    for i, g in enumerate(graphs):
        feats = {}
        feats['atom'] = torch.zeros(g.number_of_nodes('atom'), 0)  # no extra atom-level features
        feats['bond'] = torch.zeros(g.number_of_nodes('bond'), 0)  # no extra bond-level features
        feats['global'] = my_dense_feats[i].unsqueeze(0)           # NSPDK per-graph feature
        my_feats.append(feats)
    
    # Prepare reaction-level inputs
    graph_feats = [{nt: g.nodes[nt].data['ft'] for nt in batched_feats} for g in graphs]
    rxn_in_feats = [{'reac' : [graph_feats[r] for r in rxn.reacs],
                     'prod' : [graph_feats[p] for p in rxn.prods]} for rxn in rxns]
    
    # Generate reaction features
    rxn_feat_gens = rxns
    rxn_features = [gen.get_g_fts(rxn_in_feat['reac'], rxn_in_feat['prod']) 
                    for gen, rxn_in_feat in zip(rxn_feat_gens, rxn_in_feats)]
    
    # Concatenate NSPDK to final global features
    for i, feats in enumerate(rxn_features):
        feats['global'] = torch.cat([feats['global'], my_dense_feats[i].unsqueeze(0)], dim=-1)
    
    return rxn_features



def quick_test(csv_path, n_molecules=10):
    """
    Minimal test with just n_molecules
    """
    print(f"🧪 ULTRA-MINIMAL TEST: {n_molecules} molecules")
    print("=" * 60)

    # 1. Load tiny dataset
    print(f"\n1. Loading {n_molecules} molecules...")
    repo = DirectSmilesRepo()
    with open(csv_path, 'r') as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= n_molecules:
                break
            repo.append_reaction(
                [row['Parentsmiles']],
                [row['Frag1smiles'], row['Frag2smiles']],
                [int(row['BondIndex'])],
                float(row['BDH'])
            )
    print(f"   ✓ Loaded {len(repo.values)} reactions")

    # 2. Build graphs
    print("\n2. Building DGL graphs...")
    dgl_map = DGLwBDEMappings(repo)
    print(f"   ✓ Built {len(dgl_map.canon_to_dgl)} graphs")

    # 3. Test WITHOUT NSPPK
    print("\n3. Testing WITHOUT NSPPK...")
    t1 = time.time()

    atomic_nums = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17]
    featurizers_base = {
        'atom': AtomFeaturize(['atomic_num', 'total_degree'], atomic_nums),
        'bond': BondFeaturize(['is_in_ring']),
        'global': GlobalFeaturize(['num_atoms', 'num_bonds'])
    }

    dataset_base = BDEDataset.from_initials(
        repo, dgl_map,
        featurizers=featurizers_base,
        std_data=False,  # Skip standardization for speed
        load_graphs=True
    )

    t2 = time.time()
    base_dim = dataset_base.feats['global'][0].shape[1]
    print(f"   ✓ Base global features: {base_dim} dims ({t2 - t1:.2f}s)")

    # 4. Test WITH NSPPK (minimal params)
    print("\n4. Testing WITH NSPPK...")
    t1 = time.time()

    # Minimal NSPPK for speed
    nsppk_params = {
        'radius': 1,
        'distance': 1,
        'connector': 0,  # 0 = fastest
        'nbits': 3,  # 3 bits = only 8 features!
        'sigma': 1
    }

    featurizers_nsppk = {
        'atom': AtomFeaturize(['atomic_num', 'total_degree'], atomic_nums),
        'bond': BondFeaturize(['is_in_ring']),
        'global': GlobalFeaturize(['num_atoms', 'num_bonds'], nsppk_params=nsppk_params)
    }

    dataset_nsppk = BDEDataset.from_initials(
        repo, dgl_map,
        featurizers=featurizers_nsppk,
        std_data=False,  # Skip standardization for speed
        load_graphs=True
    )

    t2 = time.time()
    nsppk_dim = dataset_nsppk.feats['global'][0].shape[1]
    print(f"   ✓ NSPPK global features: {nsppk_dim} dims ({t2 - t1:.2f}s)")

    # 5. Verify
    print("\n5. Verification:")
    print(f"   Base dimension:     {base_dim}")
    print(f"   NSPPK dimension:    {nsppk_dim}")
    print(f"   Features added:     {nsppk_dim - base_dim}")
    print(f"   Expected (2^{nsppk_params['nbits']}):   {2 ** nsppk_params['nbits']}")

    if nsppk_dim - base_dim == 2 ** nsppk_params['nbits']:
        print("\n SUCCESS! NSPPK integrated correctly!")
    else:
        print("\n Dimension mismatch!")

    # 6. Show actual features
    print("\n 6. Sample features from first molecule:")
    loader = RxnDataLoader(dataset_nsppk, batch_size=1, shuffle=False)
    for (batch_data, batch_values) in loader:
        _, feats, _, idxs = batch_data
        global_feat = feats['global'][0].squeeze()
        print(f"   Global features shape: {global_feat.shape}")
        print(f"   First 5 values: {global_feat[:5].tolist()}")
        print(f"   Last 5 values:  {global_feat[-5:].tolist()}")
        break

    print("\n" + "=" * 60)
    print("Test complete!")
    print("\n Next: Try increasing nbits (3->4->5) or n_molecules (10->50->100)")
    print("=" * 60)


if __name__ == "__main__":
    # Run the test
    csv_path = 'architecture/data/train_subset.csv'

    # Start with just 10 molecules
    quick_test(csv_path, n_molecules=10)

    # Uncomment to test with more:
    # quick_test(csv_path, n_molecules=50)
