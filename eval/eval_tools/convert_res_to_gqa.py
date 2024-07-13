import os
import json
import argparse
from omegaconf import OmegaConf
from utils.util import instantiate_from_config

parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str)
parser.add_argument("--dst", type=str)
args = parser.parse_args()
cfg = OmegaConf.load('/mnt/bn/yangmin-priv/luoruipu/code/Edit-GPT4/config/datasets/eval/GQA.yaml')
ds = instantiate_from_config(cfg[0])

all_answers = []
res = json.load(open(args.src))
for line in res:
    index = int(line['item_id'].split('_')[-1])
    question_id = ds.keys[index]
    text = line['prediction'].replace('</s>', '').rstrip('.').strip().lower()
    all_answers.append({"questionId": question_id, "prediction": text})

with open(args.dst, 'w') as f:
    json.dump(all_answers, f)