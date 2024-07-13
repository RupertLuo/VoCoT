from omegaconf import OmegaConf
from utils.util import instantiate_from_config
import argparse, json
from collections import defaultdict
import pandas as pd

label2index = {'A':0, 'B': 1, 'C': 2, 'D': 3}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, default=None)
    parser.add_argument('--config', type=str, default='/mnt/bn/yangmin-priv/luoruipu/code/Edit-GPT4/config/datasets/eval/MMBench_DEV_opt.yaml')
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    mmbench_meta = pd.read_table(cfg[0]['params']['path'])
    res = json.load(open(args.result))
    for item in res:
        if 'predict' not in item:
            item['predict'] = item['prediction']
    class2res = defaultdict(list)
    if isinstance(res[0]['predict'], int):
        acc = 0
        for item in res:
            index = int(item['item_id'].split('_')[-1])
            q_type = mmbench_meta.iloc[index]['category']
            if item['predict'] == '':
                item['predict'] = 0
            if item['predict'] == label2index[item['label']]:
                c = 1
                acc += 1
            else:
                c = 0
            class2res[q_type].append(c)
        print('General accuracy: {}'.format(acc/len(res)))
        for k,v in class2res.items():
            print('{} Accuracy: {}'.format(k, sum(v)/len(v)))
        return
    
    # cfg = OmegaConf.load(args.config)
    # ds = instantiate_from_config(cfg[0])


    # item2logit = defaultdict(list)
    # item2answer = {}

    # for item in res:
    #     item_id = int(item['item_id'].split('_')[-1])
    #     question_id, option = ds.get_index(item_id)
    #     label = ds.getlabel(item_id)
    #     if question_id in item2answer:
    #         assert label == item2answer[question_id]
    #     else:
    #         item2answer[question_id] = label
    #     item2logit[question_id].append([item['logit'], option])
    
    # acc = 0
    # for k in item2logit:
    #     preds = sorted(item2logit[k], key=lambda x:x[0])[0][1]
    #     print(preds, item2answer[k])
    #     if preds == item2answer[k]:
    #         acc += 1
    # print('accuracy: {}'.format(acc/len(item2logit)))

if __name__=='__main__':
    main()