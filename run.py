# this file just makes the import system of python less painful when running scripts w/o a notebook

# import novel_arch.tests.deep_attn.bondnet_sample
from novel_arch.train_archs import deepatomsum, deepatomattn, deepatomattnnoedges, deepatomglobalattn, bondnet_original

for i in range(4):
    print('---------------- Trial {} ----------------'.format(i))
    print('------ deep atom and global attn ------')
    deepatomglobalattn()
    print('------ deep atom attn ------')
    deepatomattn()
    # print('------ deep atom attn no edges ------')
    # deepatomattnnoedges()
    # print('------ deep atom sum ------')
    # deepatomsum()
    print('------ bond net original ------')
    bondnet_original()