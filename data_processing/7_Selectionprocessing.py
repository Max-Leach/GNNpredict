import time

'''
Produces the molecule_list required to execute FFconformer and Indexed_Selection used for Analyse
'''

def dict_check(diction, key, string, value, fragtype, bondtype):
    parent_val = diction.get(key)
    if parent_val != None:
        string = string.replace(key, str(parent_val), 1)
        return value, string
    else:
        diction[key] = value #increment value afterwards
        with open('molecule_list', 'a') as listfile:
            listfile.write(str(value) + ' ' + key + ' ' + str(0) + ' ' + str(fragtype) + ' ' + bondtype + '\n')
        string = string.replace(key, str(value), 1)
        return value+1, string


if __name__ == "__main__":
    SDFdirectory = 'molecules'
    parentmoldict = {}
    fragmoldict = {}
    Index = 0
    start = time.time()
    print("Starting Selection processing")

    with open('selection.csv', 'r') as inputfile:
        #inputfile.readline()
        for line in iter(lambda: inputfile.readline(), ''): 
            arr = line.split(',')
            Index, line = dict_check(parentmoldict, arr[2], line, Index, 1, arr[6])
            Index, line = dict_check(fragmoldict, arr[3], line, Index, 2, arr[6])
            Index, line = dict_check(fragmoldict, arr[4], line, Index, 2, arr[6])
            with open('indexed_selection', 'a') as indexfile:
                indexfile.write(line.rstrip('\n') + '\n')
    end = time.time()
    print("Selection Processing complete, total elapsed time: " + str(end-start))
