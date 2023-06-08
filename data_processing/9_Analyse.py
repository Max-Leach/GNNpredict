import re
import glob
import os
import pandas as pd


#return in the form(del_E, del_H)
def Scrape(fileId):
    Energy = 0
    Enthalpy = 0

    for file in glob.glob(fileId + '_*'): filename = file
    with open(filename, 'r') as f:
        for line in f:
            if re.search('Done',line):
                Energy = re.sub('(^[^=]*=) | ( .*)','',line.rstrip('\n'))
            if re.search('Sum of electronic and thermal Enthalpies',line):
                Enthalpy = re.sub('^[^=]*= | \s','',line.rstrip('\n'))
    return float(Energy),float(Enthalpy)

def insert_energies():
    df = pd.read_csv('selection.csv')
    df.insert(6,"BDH", None)
    with open('indexed_selection', 'r') as input_file:
        for line in iter(lambda: input_file.readline(), ''): 
            values = line.split(',')
            parentvals = Scrape(values[2])
            frag1vals = Scrape(values[3])
            frag2vals = Scrape(values[4])

            BDE = frag1vals[0] + frag2vals[0] - parentvals[0]
            BDH = frag1vals[1] + frag2vals[1] - parentvals[1]

            #get BDH from last Done
            df.loc[values[0],'BDH'] = BDH
            df.loc[values[0],'BDE'] = BDE
    df.to_csv('updated_dataset.csv', index=False)

if __name__ == '__main__':
    insert_energies()
