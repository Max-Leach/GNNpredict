import dgl

## convert into graph for D-MPNN style message passing (as in Grambow)
# from HeteroMoleculeGraph shape.
# useful for neural net implementing that kind of message passing but using bondnet
# type of graph input

# note: no feature transform!

# note: consider need for self loops!
def to_directed_mpnn_g(g):
    # isolated atom case
    # if g.num_nodes('atom') == 1:
    #     return isolated_atom_case(g)
    # print("num atoms:", g.num_nodes('atom'))

    to_dbs_list = new_d_bond_map(g)
    db_count = len(to_dbs_list)

    db2db_pairs = ([],[])
    for from_db in range(db_count):
        to_dbs = to_dbs_list[from_db]
        for to_db in to_dbs:
            db2db_pairs[0].append(from_db)
            db2db_pairs[1].append(to_db)

    db2g_pairs = (tuple(range(db_count)), [0] * db_count)
    shape = {
        ('d_bond', 'db2db', 'd_bond') : db2db_pairs,
        ## one of the following may be unneeded
        # if we prevent propagation one way or another
        ('d_bond', 'db2g', 'global') : db2g_pairs,
        ('global', 'g2db', 'd_bond') : tuple(reversed(db2g_pairs)),
    }
    dg = dgl.heterograph(shape)
    return dg

# given bondnet type graph, get map of new directed bonds to other new directed bonds
# and any artifact nodes that need to be created
def new_d_bond_map(g):
    db_to_a_b = [] # directed bond it -> successor atom, old bond id associated
    b_src_to_db = {} # old bond id -> src atom -> new directed bond id
    # artifact_dbs = []
    for b in range(g.num_nodes('bond')):
        # new directed bond ids that will have dest atom as u or v
        to_u = len(db_to_a_b)
        from_u = to_u + 1

        atoms = g.predecessors(b, etype='a2b')
        # print("atoms connected!", len(atoms))
        # if len(atoms) == 1:
        #     print("attached boond:", g.predecessors(u, etype='b2a'))
        assert len(atoms) <= 2 # bond should have two member atoms
        assert len(atoms) > 0 # non-sensical bonds case
        u = atoms[0].item()
        if len(atoms) == 1:
            db_to_a_b += [(u, b), (None, b)]
            b_src_to_db[b] = {
                u: from_u,
            }
            continue
        v = atoms[1].item()
        # print("atoms:", u, v)
        db_to_a_b += [(u, b), (v, b)]
        b_src_to_db[b] = {
            # remember if d_bond is to_u, its src atom is v
            u: from_u,
            v: to_u,
        }
    
    db_to_dbs = []
    for db in range(len(db_to_a_b)):
        dest_a, b = db_to_a_b[db]
        if dest_a is None:
            db_to_dbs.append(tuple())
            continue
        out_bs = g.successors(dest_a, etype='a2b')
        out_bs = map(lambda b: b.item(), out_bs)
        out_bs = filter(lambda ab: ab != b, out_bs) # omit self loop, may remove
        to_dbs = map(lambda b: b_src_to_db[b][dest_a], out_bs)
        db_to_dbs.append(tuple(to_dbs))
    return db_to_dbs

# return dmpnn graph if g is only an atom
def isolated_atom_case(g):
    db2db_pairs = ([], [])
    db2g_pairs = ([0, 1], [0,0])
    shape = {
        ('d_bond', 'db2db', 'd_bond') : db2db_pairs,
        ## one of the following may be unneeded
        # if we prevent propagation one way or another
        ('d_bond', 'db2g', 'global') : db2g_pairs,
        ('global', 'g2db', 'd_bond') : tuple(reversed(db2g_pairs)),
    }
    dg = dgl.heterograph(shape)
    return dg