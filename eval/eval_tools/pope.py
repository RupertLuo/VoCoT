import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data')
args = parser.parse_args()
res = json.load(open(args.data))

invalid = correct = 0
for item in res:
    key = 'predict' if 'predict' in item else 'prediction'
    pred = item[key].replace('</s>', '').strip().lower()
    if pred not in ['yes', 'no']:
        if pred.startswith('yes'):
            p = 'yes'
        elif pred.startswith('no'):
            p = 'no'
        else:
            invalid += 1
            p = 'no'
        # print(pred)
        # invalid += 1
        # correct += 1 if item['label'] == 'no' else 0
    else:
        p = pred
    correct += 1 if p==item['label'] else 0

print('accuracy: {}, invalid rate: {}'.format(correct / len(res), invalid/len(res)))