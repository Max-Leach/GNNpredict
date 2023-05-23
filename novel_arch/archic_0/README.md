# Model 0
This will use components from the BonDnet paper - Chem. Sci., 2021,12, 1858-1868, and the Grambow paper - J. Phys. Chem. Lett. 2020, 11, 2992−2997.

g2g module from bondnet will use dmpnn from grambow, which will only have directed edge features.
readout will convert directed edge features into atomic features, no bond features here.