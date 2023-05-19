import dgl
import torch

from novel_arch.archic_0.directed_graph_transform import to_directed_mpnn_g

# "light" isomorphism check via checking node in and out degrees
# should show some level of isomorphism if all in and out degrees match for all nodes
def test_to_directed_mpnn_g_degree_pairing():
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
    # db2db = ([3,3, 4,4, 5,5,  0,1,2,3,4,5], [1,2, 0,2, 0,1,  0,1,2,3,4,5])
    db2db = ([3,3, 4,4, 5,5], [1,2, 0,2, 0,1]) # no self loops
    db2g = ([0,1,2,3,4,5], [0,0,0,0,0,0])
    TARGET = dgl.heterograph({
        ('d_bond', 'db2db', 'd_bond') : db2db,
        ## one of the following may be unneeded
        # if we prevent propagation one way or another
        ('d_bond', 'db2g', 'global') : db2g,
        ('global', 'g2db', 'd_bond') : tuple(reversed(db2g)),
        # self loop global
        # ('global', 'g2g', 'global') : ([0], [0])
    })

    result = to_directed_mpnn_g(ORIG)

    # perform checks that should pass for a small graph (above) if they're isomorphic
    node_check = ['d_bond', 'global']
    for n in node_check:
        assert result.num_nodes(n) == TARGET.num_nodes(n)
    edge_check = ['db2db', 'db2g', 'g2db'] #, 'g2g']
    for e in edge_check:
        assert result.num_edges(e) == TARGET.num_edges(e)
        resultDegPairs = set(zip(result.in_degrees(etype=e).tolist(), result.out_degrees(etype=e).tolist()))
        targetDegPairs = set(zip(TARGET.in_degrees(etype=e).tolist(), TARGET.out_degrees(etype=e).tolist()))
        assert resultDegPairs == targetDegPairs

# ensure multiple globals properly attached to bonds
# very important for batched graphs!
def test_to_directed_mpnn_g_global_matching():
    # original graph case is one atom with 3 others connected to it
    # 0 is the connected atom
    # bond n is connected to atom 0 and n + 1
    a2b = ([0,0,0, 1,2,3], [0,1,2, 0,1,2])
    g2b = ([0,2,1], [0,1,2]) # multiple globals
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
    # db2db = ([3,3, 4,4, 5,5,  0,1,2,3,4,5], [1,2, 0,2, 0,1,  0,1,2,3,4,5]) # last bit are self loop
    db2db = ([3,3, 4,4, 5,5], [1,2, 0,2, 0,1]) # no self loops
    db2g = ([0,1,2, 3,4,5], [0,2,1, 0,2,1]) # multiple globals
    TARGET = dgl.heterograph({
        ('d_bond', 'db2db', 'd_bond') : db2db,
        ## one of the following may be unneeded
        # if we prevent propagation one way or another
        ('d_bond', 'db2g', 'global') : db2g,
        ('global', 'g2db', 'd_bond') : tuple(reversed(db2g)),
        # self loop global
        # ('global', 'g2g', 'global') : ([0,1,2], [0,1,2])
    })

    result = to_directed_mpnn_g(ORIG)

    # perform checks that should pass for a small graph (above) if they're isomorphic
    node_check = ['d_bond', 'global']
    for n in node_check:
        assert result.num_nodes(n) == TARGET.num_nodes(n)
    edge_check = ['db2db', 'db2g', 'g2db']#, 'g2g']
    accum_assertions = True
    for e in edge_check:
        assert result.num_edges(e) == TARGET.num_edges(e)
        resultDegPairs = set(zip(result.in_degrees(etype=e).tolist(), result.out_degrees(etype=e).tolist()))
        targetDegPairs = set(zip(TARGET.in_degrees(etype=e).tolist(), TARGET.out_degrees(etype=e).tolist()))
        assert resultDegPairs == targetDegPairs

# if new graph has proper references to src atom, old bond in original graph
def test_to_directed_mpnn_g_original_refed():
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
    # 3 4 5 point to center, n-3 will be original atoms pointing from, or reversed of previous three d_bond
    # (ie all bonds pointing in)
    # db2db = ([3,3, 4,4, 5,5,  0,1,2,3,4,5], [1,2, 0,2, 0,1,  0,1,2,3,4,5])
    db2db = ([3,3, 4,4, 5,5], [1,2, 0,2, 0,1]) # no self loops
    db2g = ([0,1,2,3,4,5], [0,0,0,0,0,0])
    TARGET = dgl.heterograph({
        ('d_bond', 'db2db', 'd_bond') : db2db,
        ## one of the following may be unneeded
        # if we prevent propagation one way or another
        ('d_bond', 'db2g', 'global') : db2g,
        ('global', 'g2db', 'd_bond') : tuple(reversed(db2g)),
        # self loop global
        # ('global', 'g2g', 'global') : ([0], [0])
    })
    ## previous graph referencing
    TARGET.nodes['d_bond'].data['src_atom'] = torch.tensor(
            [0,0,0, # first three d_bonds point away from center
            1,2,3] # last three point into center, away from outsiders
        , dtype=torch.int)
    TARGET.nodes['d_bond'].data['old_bond'] = torch.tensor(
            [0,1,2, 0,1,2] # in and out triplets correspond to old bonds in same order
        , dtype=torch.int)

    result = to_directed_mpnn_g(ORIG)

    # signature of old references and degrees of all d_bonds should match
    db_edge = 'db2db'
    target_sig = set(zip(
        TARGET.in_degrees(etype=db_edge).tolist(), 
        TARGET.out_degrees(etype=db_edge).tolist(), 
        TARGET.nodes['d_bond'].data['src_atom'].tolist(), 
        TARGET.nodes['d_bond'].data['old_bond'].tolist()
    ))

    result_sig = set(zip(
        result.in_degrees(etype=db_edge).tolist(), 
        result.out_degrees(etype=db_edge).tolist(), 
        result.nodes['d_bond'].data['src_atom'].tolist(), 
        result.nodes['d_bond'].data['old_bond'].tolist()
    ))

    assert target_sig == result_sig