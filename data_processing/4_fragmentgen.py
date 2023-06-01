from rdkit import Chem
from fragment import *
import pandas as pd
import re

def fragmentall(file1, file2):
    csvcount, parentcount = 0, 0
    for line in iter(lambda: file1.readline(), ''): 
        linearr = line.split(' ')
        parentcount += 1
        for series in fragment_iterator(linearr[1]):
            if (series['fragment1'] != '') and (series['fragment2'] != ''): 
                file2.write(str(csvcount) + ',' + str(parentcount) + ',' + linearr[1] + ',' + series['fragment1'] + ',' + series['fragment2'] + ',' + '' + ','  + series['bond_type'] + ',' + linearr[2])
                csvcount += 1
                if ((csvcount % 1000000) == 0): print(f"Fragmentpairs: {csvcount}, Parents: {parentcount}")

with open('Unique_SMILES', 'r') as fl1, open('aa.csv', 'w') as fl2:
    fl2.write('Serial,Parentid,Parent,Frag1,Frag2,BDE,BondType,Heavy\n') 
    fragmentall(fl1, fl2)

