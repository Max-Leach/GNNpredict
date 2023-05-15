import dgl
import torch

from novel_arch.archic_0.directed_graph_transform import to_directed_mpnn_g

# check atleast that graph shape matches for a simple case
def test_to_directed_mpnn_g_isomorphism():
    # original graph case is one atom with 3 others connected to it
    # 0 is the connected atom
    # bond n is connected to atom 0 and n + 1
    a2b = ([0,0,0, 1,2,3], [0,1,2, 0,1,2])
    g2b = ([0,0,0], [0,1,2])
    g2a = ([0,0,0,0], [0,1,2,3])
    ORIG = dgl.heterograph({
        # these three, if they exist, are just self-loops
        ('bond', 'b2b', 'bond') : (torch.arange(3), torch.arange(3)),
        ('atom', 'a2a', 'atom') : (torch.arange(4), torch.arange(4)),
        ('global', 'g2g', 'global') : ([0],[0]),
        
        ('bond', 'b2a', 'atom') : tuple(reversed(a2b)),
        ('atom', 'a2b', 'bond') : a2b, 
        ('global', 'g2b', 'bond') : g2b,
        ('bond', 'b2g', 'global') : tuple(reversed(g2b)),
        ('global', 'g2a', 'atom') : g2a,
        ('atom', 'a2g', 'global') : tuple(reversed(g2a)),
    })

    # target case is that atoms are omitted, bonds are now directionally
    # attached to other bonds as per d-mpnn requirements
    # and all connected to global node
    # 0 1 2 d_bonds will point away from center
    # n-3 where (3 <= n <= 5) will be corresponding to n but reversed
    # (ie all bonds pointing in)
    db2db = ([3,3, 4,4, 5,5], [1,2, 0,2, 0,1])
    db2g = ([0,1,2,3,4,5], [0,0,0,0,0,0])
    TARGET = dgl.heterograph({
        ('d_bond', 'db2db', 'd_bond') : db2db,
        ## one of the following may be unneeded
        # if we prevent propagation one way or another
        ('d_bond', 'db2g', 'global') : db2g,
        ('global', 'g2db', 'd_bond') : tuple(reversed(db2g)),
    })

    result = to_directed_mpnn_g(ORIG)
    # perform checks that should pass for a small graph of above if they're isomorphic
    node_check = ['d_bond', 'global']
    for n in node_check:
        assert result.num_nodes(n) == TARGET.num_nodes(n)
    edge_check = ['db2db', 'db2g', 'g2db']
    for e in edge_check:
        assert result.num_edges(e) == TARGET.num_edges(e)
        resultDegPairs = set(zip(result.in_degrees(etype=e).tolist(), result.out_degrees(etype=e).tolist()))
        targetDegPairs = set(zip(TARGET.in_degrees(etype=e).tolist(), TARGET.out_degrees(etype=e).tolist()))
        assert resultDegPairs == targetDegPairs