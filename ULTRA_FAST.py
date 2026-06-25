"""
Ultra-minimal test script for NSPPK integration
Perfect for quick verification on a laptop
Uses only 10 molecules with minimal NSPPK parameters
"""

import csv
import time
from architecture.data.dataset import BDEDataset
from architecture.data.dataloader import RxnDataLoader
from architecture.data.initial_containers import DirectSmilesRepo, DGLwBDEMappings
from architecture.data.featurizers import AtomFeaturize, BondFeaturize, GlobalFeaturize


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
        print("\n   ✅ SUCCESS! NSPPK integrated correctly!")
    else:
        print("\n   ❌ Dimension mismatch!")

    # 6. Show actual features
    print("\n6. Sample features from first molecule:")
    loader = RxnDataLoader(dataset_nsppk, batch_size=1, shuffle=False)
    for (batch_data, batch_values) in loader:
        _, feats, _, idxs = batch_data
        global_feat = feats['global'][0].squeeze()
        print(f"   Global features shape: {global_feat.shape}")
        print(f"   First 5 values: {global_feat[:5].tolist()}")
        print(f"   Last 5 values:  {global_feat[-5:].tolist()}")
        break

    print("\n" + "=" * 60)
    print("✓ Test complete!")
    print("\nNext: Try increasing nbits (3->4->5) or n_molecules (10->50->100)")
    print("=" * 60)


if __name__ == "__main__":
    # Run the test
    csv_path = 'architecture/data/train_subset.csv'

    # Start with just 10 molecules
    quick_test(csv_path, n_molecules=10)

    # Uncomment to test with more:
    # quick_test(csv_path, n_molecules=50)