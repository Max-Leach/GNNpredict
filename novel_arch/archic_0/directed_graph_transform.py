import dgl

## convert into graph for D-MPNN style message passing (as in Grambow)
# from HeteroMoleculeGraph shape.
# useful for neural net implementing that kind of message passing but using this
# type of dataset

# consider need for self loops!
def to_directed_mpnn_g(g):

    db2db_pairs = ([0], [0])
    db2g_pairs = ([0], [2])
    shape = {
        ('d_bond', 'db2db', 'd_bond') : db2db_pairs,
        ## one of the following may be unneeded
        # if we prevent propagation one way or another
        ('d_bond', 'db2g', 'global') : db2g_pairs,
        ('global', 'g2db', 'd_bond') : tuple(reversed(db2g_pairs)),
    }
    g = dgl.heterograph(shape)

    db2db = ([3,3, 4,4, 5,5], [1,2, 0,2, 0,1])
    db2g = ([0,1,2,3,4,5], [0,0,0,0,0,0])
    return g