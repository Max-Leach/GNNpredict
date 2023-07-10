import re
import glob
import os
import pandas as pd
import numpy as np
import itertools

#return in the form(del_E, del_H)
def Scrape(directory, fileId):
    Energy1 = 0
    Energy2 = 0
    Enthalpy = 0
    globs = glob.glob(directory + fileId + '_*.log')
    if not globs: return 0,0
    else:
        for file in globs: filename = file
    try:
        with open(filename, 'r') as f:
            for line in f:

                #Field to be altered if different method is used
                if re.search('CBS-QB3 Energy',line):
                    Energy1 = re.sub('(^[^=]*=) | ( *)','',re.sub('(^[^=]*=)','',line.rstrip('\n')))
                if re.search('E\(Thermal\)',line):
                    Energy2 = re.sub('(^[^=]*=) | ( *)','',re.sub('(^[^=]*=)','',line.rstrip('\n')))
                if re.search('CBS-QB3 Enthalpy',line):
                    Enthalpy = re.sub('^\s+','',re.search('(?<==)(.*?)(?=C)',line.rstrip('\n')).group(1))
    except:
        return 0,0          
    return float(Energy1) - float(Energy2),float(Enthalpy)

#inserts the energies from log files into their respective datapoints on selection.csv based on indexed selection
#removes datapoints which could not be constructed or have an out of bounds(not in (25,175)) value
def insert_energies():
    df = pd.read_csv('selection.csv')
    pointer = -1
    with open('indexed_selection', 'r') as input_file:
        for line in iter(lambda: input_file.readline(), ''): 
            pointer += 1
            values = line.split(',')

            parentvals = Scrape('mols/', values[2])
            frag1vals = Scrape('mols/', values[3])
            frag2vals = Scrape('mols/', values[4])

            if 0 in [x for x in itertools.chain(parentvals, frag1vals,frag2vals)]: 
                continue
            else:
                BDE = (frag1vals[0] + frag2vals[0] - parentvals[0])*627.50947
                BDH = (frag1vals[1] + frag2vals[1] - parentvals[1])*627.50947

            if ((BDE < 25.0) or (BDE > 175.0)):
                continue
            #Set BDH & BDE
            df.loc[pointer,'BDH'] = float("{:.4f}".format(BDH))
            df.loc[pointer,'BDE'] = float("{:.4f}".format(BDE))
    df = df.replace(to_replace='None', value=np.nan).dropna()
    df.to_csv('updated_dataset.csv', index=False)

if __name__ == '__main__':
    insert_energies()
