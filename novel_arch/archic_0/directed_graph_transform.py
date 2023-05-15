import dgl

## convert into graph for D-MPNN style message passing (as in Grambow)
# from HeteroMoleculeGraph shape.
# useful for neural net implementing that kind of message passing but using bondnet
# type of graph input

# note: consider need for self loops!
def to_directed_mpnn_g(g):
    to_dbs_list = new_d_bond_map(g)
    db_count = len(to_dbs_list)

    db2db_pairs = ([],[])
    for from_db in range(db_count):
        to_dbs = to_dbs_list[from_db]
        for to_db in to_dbs:
            db2db_pairs[0].append(from_db)
            db2db_pairs[1].append(to_db)

    db2g_pairs = (tuple(range(db_count)), [0] * db_count)
    g2db_pairs = tuple(reversed(db2db_pairs))
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
def new_d_bond_map(g):
    db_to_a_b = [] # directed bond it -> successor atom, old bond id associated
    b_src_to_db = {} # old bond id -> src atom -> new directed bond id
    for b in range(g.num_nodes('bond')):
        # new directed bond ids that will have dest atom as u or v
        to_u = len(db_to_a_b)
        to_v = to_u + 1

        atoms = g.predecessors(b, etype='a2b')
        assert len(atoms) == 2 # bond should only have two members
        u = atoms[0].item()
        v = atoms[1].item()
        db_to_a_b += [(u, b), (v, b)]

        b_src_to_db[b] = {
            # remember if d_bond is to_u, its src atom is v
            u: to_v,
            v: to_u,
        }
    
    db_to_dbs = []
    for db in range(len(db_to_a_b)):
        dest_a, b = db_to_a_b[db]
        out_bs = g.successors(dest_a, etype='a2b')
        out_bs = map(lambda b: b.item(), out_bs)
        out_bs = filter(lambda ab: ab != b, out_bs) # omit self loop, may remove
        to_dbs = map(lambda b: b_src_to_db[b][dest_a], out_bs)
        db_to_dbs.append(tuple(to_dbs))
    return db_to_dbs