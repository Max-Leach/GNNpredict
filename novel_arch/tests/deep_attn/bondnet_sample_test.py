import pickle
import dgl

smpl = None
pathuh = '/home/pmistry/Documents/research/GNNpredict/novel_arch/bondnet_smpl'
with open(pathuh, 'rb') as smpl_f:
    smpl = pickle.load(smpl_f)

batched_graff, feats, label = smpl

# below is SOLVED!
## this gives us evidence that we need to fix the single atom case in the graph processing
# as reaction at index 0 (label presents reaction) has a single atom, if we move index it no
# longer fails in that way
# label = label[0]
# graffs = dgl.unbatch(batched_graff)
# new_graffs = [graffs[r] for r in label.reactants + label.products]
# for g in new_graffs:
#     print(g.num_nodes('atom'))
# batched_graff = dgl.batch(new_graffs)
# feats = {n : batched_graff.nodes[n].data['feat'] for n in ['atom', 'bond', 'global']}
# label = [label]

from novel_arch.deep_attn.model import DeepAtomSum

in_feat_sizes = {k : feats[k].size(-1) for k in feats}

mod = DeepAtomSum(in_feat_sizes, 32, 32, 3)
print(mod(batched_graff, feats, label).shape)