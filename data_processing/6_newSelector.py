import pandas as pd
import numpy as np
import re
import os
import random
import math
import time

'''
Select n fragments from the respective molecule bins and send the unselected molecules to a seperate directory
'''
#Sort Bond_types by smallest to largest bond_type
#Within a bond_type select based on heavy count and diversity

#globals
out_filename = 'selection.csv'
totalsize = 1800000 
bond_types = []
script_path = os.path.abspath(__file__)
script_dir = os.path.split(script_path)[0]


def ldbond_types():
    for i in os.scandir(script_dir+'/Sorted'):
        bond_types.append(re.split("sorted", i.name)[0])
    

def sample_by_cluster(df,mean, std,n_clusters = 50, Target_size=20):
    curr_size = 0
    cluster_size = int(Target_size/n_clusters)
    if (cluster_size>df.shape[0]): return df
    sample = df.sample(n=cluster_size)
    largest_std = 0

    while (curr_size<(Target_size-cluster_size)):
        cluster_arr = [df.sample(n=cluster_size) for i in range(n_clusters)]
        index = 0

        for i in range(n_clusters):
            df_concat = pd.concat((cluster_arr[i], sample))
            if i == 0:
                largest_std = sample.loc[:, 'Diversity'].std() 
                continue

            temp_std = df_concat.loc[:, 'Diversity'].std()
            if ( temp_std > largest_std):
                largest_std = temp_std
                index = i

        sample = pd.concat((cluster_arr[index], sample))
        curr_size += cluster_size
    
    return sample.drop_duplicates(subset = ['Parent', 'Frag1', 'Frag2'], keep=First)


def random_selection(filename, arguments, rangearr, distancearr, binsize):
    linenumbers = []
    sets = len(distancearr)
    for index in arguments:
        #Within here select based on diversity
        #Load n lines into array in numpy
        #Select from distribution of diversities
        df = pd.read_csv(filename,skiprows=rangearr[index],nrows=distancearr[index], header=None)
        df.columns = ['Serial','Parentid','Parent','Frag1','Frag2','BDE','BDH','BondIndex','BondType','Heavy','Diversity']

        df2 = sample_by_cluster(df, df.loc[:, 'Diversity'].mean(), df.loc[:, 'Diversity'].std(), n_clusters=15,Target_size=binsize/sets)
        selection = df2['Serial'].to_numpy()
        target = df['Serial'].to_numpy()

        for i in range(np.size(selection)):
            a = np.where(selection[i] == target)
            linenumbers.append(np.ndarray.item(a[0]) + rangearr[index])
        
        sets -= 1
    linenumbers.sort()
    return linenumbers


def recurs(arr):
    return np.argsort(np.array(arr))


def select(outputfile, totalsize):
    somecount = 0
    if not os.path.exists("Unselected"): os.makedirs("Unselected")

    file_sizes = count_lines()

    #Use recurs instead
    for element in recurs(file_sizes):
        binsize = int(totalsize/len(bond_types))
        bonds = bond_types[element]
        prev = 0
        lineNum = 1
        bounds = []
        linenumbers = []
        prev_parent = ''

        with open(script_dir + '/Sorted/' + bonds+'sorted_fragment.csv', 'r') as inputfile:
            inputfile.readline()
            #create an array of the ranges of heavy atoms
            for line in iter(lambda: inputfile.readline(), ''): 
                curr = line.split(',')[7]
                if (prev != curr): 
                    bounds.append(lineNum)
                    prev = curr
                lineNum +=1
        bounds.append(lineNum)
        
        #From the ranges of the heavy atoms calculate length of ranges, sort by smallest, then randomly select lines in those ranges to closely fit equal distribution
        distancearr = [bounds[i+1] - bounds[i] for i in range(len(bounds) - 1)]
        
        linenumbers = random_selection(script_dir + '/Sorted/' + bonds+'sorted_fragment.csv', recurs(distancearr), bounds, distancearr, binsize)

        #write selected lines to selection file and non-selected lines to
        with open(script_dir + '/Sorted/' + bonds+'sorted_fragment.csv', 'r') as inputfile, open('Unsel_' + bonds+'sorted_fragment.csv', 'w') as unselected:
            unselected.write(inputfile.readline())
            n, m = 0,0
            for line in iter(lambda: inputfile.readline(), ''):
                m += 1
                if (n == len(linenumbers)): 
                    unselected.write(line)
                    continue

                if (linenumbers[n] == m):
                    n += 1
                    curr_parent = line.split(',')[2]
                    if (prev_parent == curr_parent):
                        somecount += 1
                        continue
                    outputfile.writelines(line)
                    prev_parent = curr_parent
                else: unselected.write(line)

        os.rename(script_dir + '/' + 'Unsel_' + bonds + 'sorted_fragment.csv', script_dir + '/' + "Unselected/" + 'Unsel_' + bonds + 'sorted_fragment.csv')
        
        totalsize -= len(linenumbers) - somecount


def count_lines():
    file_size = []
    for arg in range(len(bond_types)):
        with open(script_dir + '/Sorted/' + bond_types[arg]+'sorted_fragment.csv', "r") as file:
            file_size.append(sum(1 for line in file))
    return file_size


if __name__ == "__main__":
    start = time.time()
    ldbond_types()

    #Selects from bins
    with open(out_filename, 'a') as output: select(output, totalsize)

    end = time.time()
    print("complete, total elapsed time was " + str(end-start))

