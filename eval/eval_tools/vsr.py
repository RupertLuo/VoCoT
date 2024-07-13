import argparse
from collections import defaultdict
import json
from omegaconf import OmegaConf
from utils.util import instantiate_from_config

parser = argparse.ArgumentParser()
parser.add_argument('--data')
parser.add_argument('--config')

args = parser.parse_args()
res = json.load(open(args.data))
cfg = OmegaConf.load(args.config)
ds = instantiate_from_config(cfg[0], reload=True)

invalid = correct = 0
relation2res = defaultdict(list)
if 'predict' in res[0] and isinstance(res[0]['predict'], int):
    for item in res:
        item_id = int(item['item_id'].split('_')[-1])
        relation = ds.meta[item_id]['relation']
        pred = item['predict']
        correct += 1 if pred==item['label'] else 0
        relation2res[relation].append(1 if pred==item['label'] else 0)
else:
    for item in res:
        item_id = int(item['item_id'].split('_')[-1])
        relation = ds.meta[item_id]['relation']
        key = 'predict' if 'predict' in item else 'prediction'
        pred = item[key].replace('</s>', '').strip().lower()
        if pred not in ['yes', 'no']:
            # p = -1
            if pred.startswith('yes'):
                p = 1
            elif pred.startswith('no'):
                p = 0
            else:
                invalid += 1
                p = 0
        else:
            if pred == 'yes':
                p = 1
            else:
                p = 0
        correct += 1 if p==item['label'] else 0
        relation2res[relation].append(1 if p==item['label'] else 0)

print('accuracy: {}, invalid rate: {}'.format(correct / len(res), invalid/len(res)))
print('====results in detail=====')
for k in sorted(relation2res.keys()):
    v = relation2res[k]
    print('{}: {} in {} samples'.format(k, round(sum(v)/len(v),4), len(v)))