import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default=None)

args = parser.parse_args()
res = json.load(open(args.data))
print(len(res))