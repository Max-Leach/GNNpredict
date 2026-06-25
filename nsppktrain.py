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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel

df_train = pd.read_csv("C:/Users/Maxim/Documents/KernelBDE/data/train_subset.csv")
df_test = pd.read_csv("C:/Users/Maxim/Documents/KernelBDE/data/test_subset.csv")

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

#sample_sizes = [25000, 50000, 100000, 250000, 500000, 750000, len(df_train) + len(df_test)]
sample_size = [1, 10, 50, 100, 250, 1000]

samples_train = {size: df_train.sample(n=min(int(size), len(df_train)), random_state=42) for size in sample_sizes}


X_test = df_test['Parentsmiles'].apply(smiles_to_graph)
y_test = subset_test['BDH']


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
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

#    with open(f"model_{n}_samples.pkl", "wb") as f:
#        pickle.dump(pipeline, f)

    t2 = time.time()
    print(f"Samples: {n},  MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, R2: {test_r2:.4f}, Time: {t2-t1:.2f}s\n")

#    with open("model_CV_performance.txt", "a") as f:
#        f.write(f"Samples: {n},  MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, R2: {test_r2:.4f}, Time: {t2-t1:.2f}s\n")
