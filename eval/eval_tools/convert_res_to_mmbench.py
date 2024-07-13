import os
import json
import argparse
import pandas as pd
from omegaconf import OmegaConf

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--tgt", type=str, required=True)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    df = pd.read_table(args.annotation_file)

    all_answers = []
    res = json.load(open(args.src))
    cur_df = df.copy()
    cur_df = cur_df.drop(columns=['hint', 'category', 'source', 'image', 'comment', 'l2-category'])
    cur_df.insert(6, 'prediction', None)
    for pred in res:
        cur_df.loc[df['index'] == pred['dataset_id'], 'prediction'] = pred['prediction'].replace('</s>', '')

    cur_df.to_excel(args.tgt, index=False, engine='openpyxl')