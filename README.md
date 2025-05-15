<div align="center">
<h2>DeepBDE: whatever else is in the paper title</h2>

</div>

---

Development of quantum algorithms for bridging the gap between quantum chemistry and deep learning via GNNs et al.

## Getting Started

1. **Setup Environment**

    Create an environment with all necessary dependencies. This can be done using Conda:

    ```bash
    conda create -n "deepbde" python=3.12
    conda activate deepbde
    pip install -r requirements.txt
    ```

2. **Download model and transforms (or dataset CSV if training)**

### To run predictions

3. **Extract model and transform**

    Place these files in the parent directory of this repo.

4. **Inferencing**

    **Single reaction inferencing** - requires reactant SMILES and bond index. Bond index is defined by how RDkit arranges bond order in the molecule.

    Example: split reactant given by SMILES: CCOc1cccc(O)c1 at bond index 1.
    
    Products are: [O]c1cccc(O)c1	[CH2]C

    ```bash
    python infer.py [O]c1cccc(O)c1 1
    ```

    Same as above, but use products to cross-check that the reaction products are the same as you expect (if not, an error is generated).

    ```bash
    python infer.py [O]c1cccc(O)c1 1 [O]c1cccc(O)c1 [CH2]C
    ```

## Training

3. **Create encoded dataset**

4. **Train model given hyperparameters**

should we include training or ditch it for the public repo?