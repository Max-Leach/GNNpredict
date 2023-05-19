import dgl
import torch

## convert into graph for D-MPNN style message passing (as in Grambow)
# from HeteroMoleculeGraph shape.
# useful for neural net implementing that kind of message passing but using bondnet
# type of graph input
# -note that this function doesn't do a feature transform

# note: consider need for self loops!
def to_directed_mpnn_g(g):
    # isolated atom case
    # if g.num_nodes('atom') == 1:
    #     return isolated_atom_case(g)
    # print("num atoms:", g.num_nodes('atom'))

    to_dbs_list, b_src_to_db, db_to_g = new_d_bond_map(g)
    db2db_pairs = expand_map_to_edge_pairs(to_dbs_list)
    db2g_pairs = expand_map_to_edge_pairs(db_to_g)

    shape = {
        ('d_bond', 'db2db', 'd_bond') : db2db_pairs,
        ## one of the following may be unneeded
        # if we prevent propagation one way or another
        ('d_bond', 'db2g', 'global') : db2g_pairs,
        ('global', 'g2db', 'd_bond') : tuple(reversed(db2g_pairs)),
        ('global', 'g2g', 'global') : ([], []) # for adding self loop later
    }
    dg = dgl.heterograph(shape)
    glob_count = dg.num_nodes('global')
    # dg = dgl.add_edges(dg, range(glob_count), range(glob_count), etype=('global', 'g2g', 'global'))
    # dg = dgl.add_self_loop(dg, etype=('global', 'g2g', 'global'))

    ## referencing old graph here
    src_atoms_for_db = torch.zeros([len(to_dbs_list)], dtype=torch.int)
    old_bonds_for_db = torch.zeros([len(to_dbs_list)], dtype=torch.int)
    for b in b_src_to_db:
        bond_db = b_src_to_db[b]
        for src_atom in bond_db:
            db = bond_db[src_atom]
            src_atoms_for_db[db] = src_atom
            old_bonds_for_db[db] = b

    dg.nodes['d_bond'].data['src_atom'] = src_atoms_for_db
    dg.nodes['d_bond'].data['old_bond'] = old_bonds_for_db

    return dg

# expand a list from below mapping function into pairs that are accepted into heterograph construction
def expand_map_to_edge_pairs(mapping):
    srcs, dests = [], []
    for src in range(len(mapping)):
        targets = mapping[src]
        for dest in targets:
            srcs.append(src)
            dests.append(dest)
    return srcs, dests

# given bondnet type graph, get map of new directed bonds to other new directed bonds
# and any artifact nodes that need to be created
def new_d_bond_map(g):
    db_to_a_b = [] # directed bond it -> successor atom, old bond id associated
    b_src_to_db = {} # old bond id -> src atom -> new directed bond id
    db_to_g = [] # directed bond -> global(s) attached to (same indexing as db_to_a_b)
    # artifact_dbs = []
    for b in range(g.num_nodes('bond')):
        # new directed bond ids that will have dest atom as u or v
        to_u = len(db_to_a_b)
        from_u = to_u + 1

        atoms = g.predecessors(b, etype='a2b')
        assert len(atoms) <= 2 # bond should have two member atoms
        assert len(atoms) > 0 # non-sensical bonds case if false
        u = atoms[0].item()
        globs = g.predecessors(b, etype='g2b')

        if len(atoms) == 1:
            db_to_a_b += [(u, b), (None, b)]
            b_src_to_db[b] = {
                u: from_u,
            }
            db_to_g += [globs]
            
            continue
    
        v = atoms[1].item()
        db_to_a_b += [(u, b), (v, b)]
        b_src_to_db[b] = {
            # remember if d_bond is to_u, its src atom is v
            u: from_u,
            v: to_u,
        }
        db_to_g += [globs, globs]

    
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
    return db_to_dbs, b_src_to_db, db_to_g

# return dmpnn graph for if g is only an atom
def isolated_atom_case(g):
    assert False # function may not be complete, globals improperly attached?
    db2db_pairs = ([], [])
    db2g_pairs = ([0,1], [0,0])
    shape = {
        ('d_bond', 'db2db', 'd_bond') : db2db_pairs,
        ## one of the following may be unneeded
        # if we prevent propagation one way or another
        ('d_bond', 'db2g', 'global') : db2g_pairs,
        ('global', 'g2db', 'd_bond') : tuple(reversed(db2g_pairs)),
    }
    dg = dgl.heterograph(shape)
    ## referencing old graph here
    dg.nodes['d_bond'].data['src_atom'] = torch.tensor([0,0])
    dg.nodes['d_bond'].data['old_bond'] = torch.tensor([0,0])

    return dg