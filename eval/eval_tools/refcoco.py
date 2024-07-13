import argparse
import json
from utils.eval_util import *
import tqdm
from functools import partial

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=None)
    parser.add_argument('--method', type=str, default='str')
    parser.add_argument('--mistral', action='store_true')
    args = parser.parse_args()

    result = json.load(open(args.path))
    if args.method == 'str':
        extract_fn = partial(extract_box_str, mistral=args.mistral)
    elif args.method == 'llava':
        extract_fn = extract_box_str_llava
    elif args.method == 'llava16':
        extract_fn = partial(extract_box_str_llava16, mistral=args.mistral)
    elif args.method == 'space':
        extract_fn = extract_box_str_space
    elif args.method == 'special_tokens':
        extract_fn = extract_box_str
    elif args.method == 'qwenvl':
        extract_fn = extract_box_str_qwenvl
    elif args.method == 'minigptv2':
        extract_fn = extract_box_str_minigptv2
    else:
        raise NotImplementedError
    samples = {
        'fail': 0,
        'wrong': 0,
        'correct': 0
    }
    key = 'prediction' if 'prediction' in result[0] else 'predict'
    for item in result:
        # print(item)
        box = extract_fn(item[key][0] if isinstance(item[key], list) else item[key])
        if box is None:
            print(item[key])
            samples['fail'] += 1
        else:
            iou = cal_iou(box, item['label'])
            if iou >= 0.5:
                samples['correct'] += 1
            else:
                samples['wrong'] += 1
    print(json.dumps(samples)) 
    print('accuracy: {}'.format(samples['correct'] / len(result)))

if __name__=='__main__':
    main()