from novel_arch.deep_attn.data.dset_generate import from_csv
from statistics import mean, stdev

def print_stats():
    dset = from_csv('/home/pmistry/Documents/research/data/ALFABET_data/acp_updated_NoDupes.csv', max_lines=800, start_line=1)

    vals = [item[1] for item in dset]
    print('ur so mean', mean(vals))
    print('stdev', stdev(vals))