import json,argparse
from utils.eval_util import extract_box, remove_all_box_str

parser = argparse.ArgumentParser()
parser.add_argument('--src', type=str)
parser.add_argument('--tgt', type=str)
parser.add_argument('--desc', action='store_true')
args = parser.parse_args()

res = json.load(open(args.src))
new_res = []
if args.desc:
    for item in res:
        key = 'prediction' if 'prediction' in item else 'predict'
        pred = remove_all_box_str(item[key], mistral=True).replace('  ', ' ').replace('</s>', '').strip().replace(' .', '.')
        new_res.append({
        'id': item['label'],
        'response': pred
    })
else:
    for item in res:
        tmp = item['prediction'].replace('</s>', '').strip().lower()
        if 'yes' in tmp:
            pred = 'Yes'
        else:
            pred = 'No'
        new_res.append({
            'id': item['label'],
            'response': pred
        })
with open(args.tgt, 'w') as wf:
    json.dump(new_res, wf) 