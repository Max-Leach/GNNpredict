from full_pipeline import dataset_and_split
import argparse

parser = argparse.ArgumentParser(description='create a DeepBDE dataset lazy_loaded from a csv file')

parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--csv_path', type=str, required=True)

args = parser.parse_args()

dataset_and_split.do(args.save_dir, args.csv_path, [0.8, 0.1, 0.1])