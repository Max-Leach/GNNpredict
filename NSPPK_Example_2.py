# example2.py

import ast
from scipy.sparse import hstack, issparse, csr_matrix
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
from multiprocessing import Pool
import time
import gc
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import pandas as pd
import networkx as nx
import numpy as np

from scipy.sparse import vstack, csr_matrix

from rdkit import Chem
from nsppk import NodeNSPPK

from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize


RANDOM_STATE = 42

def to_numpy(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def standardize_train_test(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s


def eval_regression(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    print(f"{label:15s} | MAE: {mae:.6f} | MSE: {mse:.6f}")
    return {"model": label, "mae": mae, "mse": mse}


def smiles_to_graph_arrays(cann):
    mol = Chem.AddHs(Chem.MolFromSmiles(cann))

    n_atoms = mol.GetNumAtoms()

    # Node features
    X = np.zeros((n_atoms, 8), dtype=float)
    for i, atom in enumerate(mol.GetAtoms()):
        X[i] = [
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetMass(),
            int(atom.GetHybridization()),
            atom.GetNumExplicitHs(),
            atom.GetDegree(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing())
        ]

    # Edge index
    edges = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    edge_index = np.array(edges).T  # shape (2, E)

    return X, edge_index

def smiles_to_graph_fast(cann):
    mol = Chem.AddHs(Chem.MolFromSmiles(cann))
    G = nx.Graph()

    # --- Nodes ---
    nodes = []
    for atom in mol.GetAtoms():
        nodes.append((
            atom.GetIdx(),
            {
                'label': atom.GetSymbol(),
                'vec': np.array([
                    atom.GetAtomicNum(),
                    atom.GetFormalCharge(),
                    atom.GetMass(),
                    int(atom.GetHybridization()),
                    atom.GetNumExplicitHs(),
                    atom.GetDegree(),
                    int(atom.GetIsAromatic()),
                    int(atom.IsInRing())
                ], dtype=float)
            }
        ))

    G.add_nodes_from(nodes)

    # --- Edges ---
    edges = []
    for bond in mol.GetBonds():
        edges.append((
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            {'label': str(bond.GetBondType())}
        ))

    G.add_edges_from(edges)

    return G

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


def load_data():
    df_train = pd.read_csv("nsppk_data/train_subset.csv")
    df_test = pd.read_csv("nsppk_data/test_subset.csv")
    df_val = pd.read_csv("nsppk_data/val_subset.csv")

    #df_train = df_train[:10000]
    #df_test = df_test[:10000]
    #df_val = df_val[:10000]

    df_train = df_train.iloc[:1000].reset_index(drop=True)
    df_test = df_test.iloc[:1000].reset_index(drop=True)
    df_val = df_val.iloc[:1000].reset_index(drop=True)


    #X_train = df_train['Parentsmiles'].apply(smiles_to_graph)
    #y_train = df_train['BDH']

    t1 = time.time()
    print("Starting X_train")
    with Pool(processes=4) as p:
        X_train_ = list(p.map(smiles_to_graph_fast, df_train['Parentsmiles'], chunksize=50))

    t2 = time.time()
    print(f"X_train finished.. in {t2-t1} seconds")

    X_train = X_train_[:1000]

    del X_train_
    gc.collect()

    t1 = time.time()
    print("Starting X_test")
    with Pool(processes=4) as p:
        X_test_ = list(p.map(smiles_to_graph_fast, df_test['Parentsmiles'], chunksize=50))

    t2 = time.time()
    print(f"X_test finished.. in {t2 - t1} seconds")

    X_test = X_test_[:1000]

    del X_test_
    gc.collect()


    ##
    t1 = time.time()
    print("Starting X_val")
    with Pool(processes=4) as p:
        X_val_ = list(p.map(smiles_to_graph_fast, df_val['Parentsmiles'], chunksize=50))

    t2 = time.time()
    print(f"X_val finished.. in {t2 - t1} seconds")

    X_val = X_val_[:1000]

    del X_val_
    gc.collect()

    y_train = df_train['BDH']
    y_train=y_train[:1000]

    y_test = df_test['BDH']
    y_test=y_test[:1000]

    y_val = df_val['BDH']
    y_val = y_val[:1000]

    return df_train, df_test, df_val, X_train, y_train, X_test, y_test, X_val, y_val


def parse_embedding_cell(x):
    """
    Converts Embeddings cell to 1D numpy array.
    Handles:
    - list object
    - string like "[0.1, 0.2,...]"
    """
    if isinstance(x, list):
        arr = np.asarray(x, dtype=float)
    elif isinstance(x, str):
        arr = np.asarray(ast.literal_eval(x), dtype=float)
    else:
        raise ValueError(f"Unsupported embedding cell type: {type(x)}")
    return arr.reshape(-1)


def load_gnn_embeddings_df(path):
    df = pd.read_csv(path)
    if "Embeddings" not in df.columns:
        raise ValueError(f"'Embeddings' column not found in {path}")
    df["Embeddings"] = df["Embeddings"].apply(parse_embedding_cell)
    return df

def sparse_safe_concat(A, B):
    if issparse(A) or issparse(B):
        if not issparse(A):
            A = csr_matrix(A)
        if not issparse(B):
            B = csr_matrix(B)
        return hstack([A, B]).tocsr()
    return np.concatenate([A, B], axis=1)


def main():
    # -------------------------------
    # 1) Load data
    # -------------------------------
    print("Attempting to load data.")
    df_train, df_test, df_val, X_train, y_train, X_test, y_test, X_val, y_val = load_data()
    print(f"Data loaded. {df_train.shape}")

    print(df_train['Parentsmiles'].iloc[0])
    print(df_train['Parentsmiles'].iloc[1])
    # -------------------------------
    # 2) GNN/GIN features
    # -------------------------------
    print("Extracting embeddings...")

    X_gnn_train_df = load_gnn_embeddings_df("nsppk_data/train_out_best.csv")
    X_gnn_test_df = load_gnn_embeddings_df("nsppk_data/test_out_best.csv")
    X_gnn_val_df = load_gnn_embeddings_df("nsppk_data/val_out_best.csv")

    X_gnn_train = X_gnn_train_df["Embeddings"]
    X_gnn_test = X_gnn_test_df["Embeddings"]
    X_gnn_val = X_gnn_val_df["Embeddings"]

    X_gnn_train = np.vstack(X_gnn_train)
    X_gnn_test = np.vstack(X_gnn_test)
    X_gnn_val = np.vstack(X_gnn_val)

    print("Ok..")

    Xg_train = X_gnn_train
    Xg_test = X_gnn_test
    Xg_val = X_gnn_val

    print("Example of gnn train first one")
    print(X_gnn_train[0])

    print(f"GNN train shape: {X_gnn_train.shape}")
    print(f"GNN test shape:  {X_gnn_test.shape}")
    print(f"GNN val shape:  {X_gnn_val.shape}")
    print("GNN feats loaded!")
    # -------------------------------
    # 3) NSPPK features
    # -------------------------------
    print("Extracting NSPPK features...")
    best_params = {'radius': 1, 'distance': 1, 'connector': 1, 'nbits': 12, 'sigma': 1}

    nsppk_X_train_features = extract_features(X_train, df_train, **best_params)
    nsppk_X_test_features = extract_features(X_test, df_test, **best_params)
    nsppk_X_val_features = extract_features(X_val, df_val, **best_params)

    print("Finished NSPPK feature extraction")
    print(f"shape of nsspk feats: {nsppk_X_train_features.shape}")
    print("Done.")

    Xn_train = nsppk_X_train_features
    Xn_test = nsppk_X_test_features
    Xn_val = nsppk_X_val_features


    ### Normalise
    Xg_train_norm = normalize(Xg_train, norm='l2', axis=1)
    Xn_train_norm = normalize(Xn_train, norm='l2', axis=1)

    Xg_test_norm = normalize(Xg_test, norm='l2', axis=1)
    Xn_test_norm = normalize(Xn_test, norm='l2', axis=1)

    Xg_val_norm = normalize(Xg_val, norm='l2', axis=1)
    Xn_val_norm = normalize(Xn_val, norm='l2', axis=1)
    # -------------------------------
    # 4) Train Models
    # -------------------------------
    xgb_params = {
        'n_estimators': 5000,
        'learning_rate': 0.03,
        'max_depth': 5,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'colsample_bylevel': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 5,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }

    rf_params = {
        'n_estimators' : 200,
        'max_depth' : 7,
        'min_samples_split' : 5,
        'min_samples_leaf' : 2,
        'max_features' : 'sqrt',
        'random_state' : RANDOM_STATE,
        'n_jobs' : -1
    }


    print("Training GNN-only...")
    t1 = time.time()

    reg_gnn = XGBRegressor(**xgb_params, early_stopping_rounds=50)
    reg_gnn.fit(
        Xg_train_norm, y_train,
        eval_set=[(Xg_val_norm, y_val)],
        verbose=False
    )

    pred_gnn = reg_gnn.predict(Xg_test_norm)
    t2 = time.time()
    print(f"Time taken: {t2-t1} seconds\n")

    print("Training NSPPK-only...")
    t1 = time.time()

    reg_nsppk = XGBRegressor(**xgb_params, early_stopping_rounds=50)

    reg_nsppk.fit(
        Xn_train_norm, y_train,
        eval_set=[(Xn_val_norm, y_val)],
        verbose=False
    )

    pred_nsppk = reg_nsppk.predict(Xn_test_norm)
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds\n")
    # -------------------------------
    # 5) Fusion method
    # -------------------------------
    print("Training Concatenated (GNN + NSPPK)...")

    # Concatenate

    alpha = 0.7
    X_fused_train = sparse_safe_concat(alpha * Xg_train_norm, (1 - alpha) * Xn_train_norm)
    X_fused_test = sparse_safe_concat(alpha * Xg_test_norm, (1 - alpha) * Xn_test_norm)
    X_fused_val = sparse_safe_concat(alpha * Xg_val_norm, (1 - alpha) * Xn_val_norm)


    t1 = time.time()
    #reg_fused = RandomForestRegressor(**rf_params)
    reg_fused = XGBRegressor(**xgb_params, early_stopping_rounds=50)

    reg_fused.fit(
        X_fused_train, y_train,
        eval_set=[(X_fused_val, y_val)],
        verbose=False
    )

    pred_fused = reg_fused.predict(X_fused_test)
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds\n")

    # -------------------------------
    # 6) Results
    # -------------------------------
    print("\n=== Final Performance Comparison ===")
    results = []
    results.append(eval_regression(y_test, pred_gnn, "GNN-only"))
    results.append(eval_regression(y_test, pred_nsppk, "NSPPK-only"))
    results.append(eval_regression(y_test, pred_fused, "Concatenated"))

    res_df = pd.DataFrame(results).sort_values("mae")
    print("\n Ordered: ")
    print(res_df.to_string(index=False))

    best_standalone = min(results[0]['mae'], results[1]['mae'])

if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn")
    main()