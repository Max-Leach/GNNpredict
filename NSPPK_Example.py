# example2.py

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
    #df_validate = pd.read_csv("C:/Users/Maxim/Documents/KernelBDE/data/val_subset.csv")

    df_train = df_train[:1000]
    df_test = df_test[:1000]

    X_train = df_train['Parentsmiles'].apply(smiles_to_graph)
    y_train = df_train['BDH']

    X_test = df_test['Parentsmiles'].apply(smiles_to_graph)
    y_test = df_test['BDH']

    return df_train, df_test, X_train, y_train, X_test, y_test

def main():
    # -------------------------------
    # 1) Load data
    # -------------------------------
    print("Attempting to load data.")
    df_train, df_test, X_train, y_train, X_test, y_test = load_data()
    print(f"Data loaded. {df_train.shape}")
    # -------------------------------
    # 2) GNN/GIN features
    # -------------------------------
    print("Extracting embeddings...")
    #...
    #X_gnn_train...
    #X_gnn_test...

    # -------------------------------
    # 3) NSPPK features
    # -------------------------------
    print("Extracting NSPPK features...")
    best_params = {'radius': 1, 'distance': 1, 'connector': 1, 'nbits': 13, 'sigma': 1}

    nsppk_X_train_features = extract_features(X_train, df_train, **best_params)
    nsppk_X_test_features = extract_features(X_test, df_test, **best_params)

    print("Finished NSPPK feature extraction")
    print(f"shape of nsspk feats: {nsppk_X_train_features.shape}")
    print("Done.")
    # -------------------------------
    # 4) Concatenate features
    # -------------------------------
    #X_fused_train = np.concatenate([X_gnn_train, nsppk_X_train_features], axis=1)
    #X_fused_test = np.concatenate([X_gnn_test, nsppk_X_test_features], axis=1)

    # -------------------------------
    # 5) Train/evaluate same regressor
    # -------------------------------
    reg_gnn = RandomForestRegressor(
        n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
    )
    reg_nsppk = RandomForestRegressor(
        n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
    )
    reg_fused = RandomForestRegressor(
        n_estimators=500, random_state=RANDOM_STATE, n_jobs=-1
    )


    #reg_gnn.fit(X_gnn_train, y_train)
    #reg_nsppk.fit(nsppk_X_train_features, y_train)
    #reg_fused.fit(X_fused_train, y_train)

    #pred_gnn = reg_gnn.predict(X_gnn_test)
    #pred_nsppk = reg_nsppk.predict(nsppk_X_test_features)
    #pred_fused = reg_fused.predict(X_fused_test)

    print("\n=== Test Performance ===")
    #results = []
    #results.append(eval_regression(y_test, pred_gnn, "GNN-only"))
    #results.append(eval_regression(y_test, pred_nsppk, "NSPPK-only"))
    #results.append(eval_regression(y_test, pred_fused, "GNN+NSPPK"))

    #res_df = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
    #print("\n=== Ranked by MAE (lower is better) ===")
    #print(res_df.to_string(index=False))

    # improvement vs best standalone
    #standalone = res_df[res_df["model"].isin(["GNN-only", "NSPPK-only"])].iloc[0]
    #fused = res_df[res_df["model"] == "GNN+NSPPK"].iloc[0]

    #mae_gain = standalone["mae"] - fused["mae"]
    #mse_gain = standalone["mse"] - fused["mse"]

    #print("\n=== Fusion Gain vs Best Standalone ===")
    #print(f"Best standalone: {standalone['model']}")
    #print(f"MAE gain: {mae_gain:+.6f}")
    #print(f"MSE gain: {mse_gain:+.6f}")

    #out_path = "example2_metrics.csv"
    #res_df.to_csv(out_path, index=False)
    #print(f"\nSaved metrics: {out_path}")


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn")
    main()