from architecture.data import create_dataset_and_split
import argparse
import json

parser = argparse.ArgumentParser(description='Create a DeepBDE encoded dataset with their train, validation, test index splits, lazily loaded so you won\'t consume all your RAM with BDE reactions')

parser.add_argument('--save_dir', type=str, required=True, help='save directory to dump generated dataset and subset index splits')
parser.add_argument('--csv_path', type=str, required=True, help='path to csv to generate encoded dataset')
parser.add_argument('--split', type=json.loads, required=True, help='train, validation, test fraction split in list form (i.e. [0.8,0.1,0.1] for typical 8:1:1 split)')

args = parser.parse_args()

create_dataset_and_split.do(args.save_dir, args.csv_path, args.split)