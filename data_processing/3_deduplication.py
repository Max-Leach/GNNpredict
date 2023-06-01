import pandas as pd
import numpy as np
import re

df = pd.read_csv('acp_updated_NoDupes.csv')
count = 0

df = df.drop_duplicates(subset=['Parent'], keep='first')
arr = df['Parent'].to_numpy()

#Iff molecule exists within acp_dataset, do not include
with open('Processed_CID-SMILES', 'r') as fl1, open('Unique_SMILES', 'w') as fl2: 
    for line in iter(lambda: fl1.readline(), ''):
        curr_line = line.split(' ')[1]
        count +=1
        if (count % 1000000 == 0): print(count)
        if (np.any(arr == curr_line)): continue 

        fl2.writelines(line)

print("Complete")

