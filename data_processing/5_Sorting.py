import pandas as pd
import numpy as np
import os
import random
import time

#globals
int_filename = 'Non_iso_with_Heavy.csv'
endline = 168000000 #end of file 
steps = 10000000
illegal = ['Cl-Cl', 'Cl-H', 'Cl-F', 'F-F', 'H-H', 'F-H']
bond_types = []

script_path = os.path.abspath(__file__)
script_dir = os.path.split(script_path)[0]

#Reads n rows of intermediate
def loadrows(filename, start=0, numrows=steps):
    return pd.read_csv(filename,skiprows=start,nrows=numrows)

#Outputs bond types to respective csv file
def writetofile(df):
    df.columns = ['Serial','Parent', 'Pid', 'Frag1', 'Frag2', 'BDE', 'BondType', 'Heavy']
    df = df.drop_duplicates(subset=['Parent', 'Frag1', 'Frag2'], keep='first')
    grp = df.groupby(df.BondType)
    variants = df['BondType'].unique()
    for type in variants:
        if type in illegal: continue
        if type not in bond_types: bond_types.append(type)
        grp.get_group(type).to_csv('temp.csv', index=False)
        with open(type+'fragment.csv', 'a') as origin, open('temp.csv', 'r') as subord:
            index = 0
            for fline in iter(lambda: subord.readline(), ''): 
                if (origin.tell() != 0 and index==0): 
                    index += 1
                    continue
                origin.write(fline)

#Handles entire file sorting
def sortbybonds(filename, endcount, rowcount):
    n = 0
    while (n < endcount):
        writetofile(loadrows(filename, n, rowcount))
        n+= rowcount


#Orders csv files by heavy atom_count
def Order():
    if not os.path.exists("Sorted"): os.makedirs("Sorted")
    for variant in bond_types:
        df = pd.read_csv(variant+'fragment.csv')
        df = df.sort_values(by=['Heavy'])
        df.to_csv(script_dir + '/Sorted/' + variant+'sorted_fragment.csv', index=False)
        os.remove(variant+'fragment.csv')

if __name__ == "__main__":
    start = time.time()

    #outputs into file based on bond type
    sortbybonds(int_filename, endline, steps)

    #Sorts the file in order of bond type
    Order()
    end = time.time()
    os.remove('temp.csv')

    print("Sorting done, now selecting, section elapsed time was " + str(end - start))


