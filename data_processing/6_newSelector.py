import pandas as pd
import numpy as np
import re
import os
import random
import time

#globals
out_filename = 'selection.csv'
totalsize = 3500000
bond_types = []
script_path = os.path.abspath(__file__)
script_dir = os.path.split(script_path)[0]

def ldbond_types():
    for i in os.scandir(script_dir+'/Sorted'):
        bond_types.append(re.split("sorted", i.name)[0])
    

def random_selection(arguments, rangearr, distancearr, binsize):
    linenumbers = []
    sets = len(distancearr)
    for index in arguments:
        if (binsize/(sets) > (distancearr[index])):
            linenumbers += range(rangearr[index], rangearr[index]+distancearr[index])
            binsize -= distancearr[index]
        else:
            linenumbers += random.sample(range(rangearr[index], rangearr[index] + distancearr[index]), int(binsize/sets))
            binsize -= int(binsize/sets)
        sets -= 1
    linenumbers.sort()
    return linenumbers

def recurs(distancearr):
    return np.argsort(np.array(distancearr))

def Select(outputfile):
    binsize = totalsize/len(bond_types)
    somecount = 0
    if not os.path.exists("Unselected"): os.makedirs("Unselected")

    for bonds in bond_types:
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
        
        linenumbers = random_selection(recurs(distancearr), bounds, distancearr, binsize)

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
        
    print(somecount)

if __name__ == "__main__":
    start = time.time()
    ldbond_types()

    #Selects from bins
    with open(out_filename, 'a') as output: Select(output)

    end = time.time()
    print("complete, total elapsed time was " + str(end-start))

