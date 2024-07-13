import torch
import json, os
from PIL import Image
from utils.eval_util import cal_nn_iou, draw_all_box, colors, draw_all_box_colored

def box2str(box, mode='special_tokens', prec=2, space=False):
    if mode == 'text':
        # using text to represent the box
        if space:
            sep = ', '
        else:
            sep = ','
        tmp_format = sep.join(['{' + ':.{}f'.format(prec)+'}']*4)
        a_box = [float(o) for o in box]
        return tmp_format.format(*a_box)
    else:
        raise NotImplementedError

def allbox2str(objects, colored=False):
    s = []
    for i,obj in enumerate(objects):
        if colored:
            s.append('{}: {} box [{}]'.format(obj['class'], colors[i], box2str(obj['bbox'], 'text', 3, True)))
        else:
            s.append('{}: [{}]'.format(obj['class'], box2str(obj['bbox'], 'text', 3, True)))
    return ', '.join(s)

def filter_box(box_list, iou = 0.5):
    '''
    box_list = []
    '''
    box_tensor = []
    for i in range(len(box_list)):
        box_tensor.append(list(map(float,box_list[i]['bbox'])))
    iou_matrix = cal_nn_iou(box_tensor)
    result_box = []
    for i in range(len(iou_matrix)):
        flag = 0
        for j in result_box:
            if iou_matrix[i,j]> iou:
                flag = 1
                break
        if flag==0:
            result_box.append(i)
    result_box = [box_list[i] for i in result_box]
    return result_box

import random
from collections import defaultdict
def merge_object(info):
    class2box = defaultdict(list)
    for obj in info:
        class2box[obj['class']].append([float(t) for t in obj['bbox']])
    return class2box

class OpenImagesSource:
    def __init__(self, path='/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/OpenImages/balance_sample_merge_filter_box.json', image_base='/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/OpenImages/', draw_box=False):
        self.meta = json.load(open(path))
        self.image_base = image_base
        self.draw_box = draw_box
    
    def __len__(self,):
        return len(self.meta)

    def __getitem__(self, index):
        item = self.meta[index]
        split_name = 'train_{}'.format(item['image'][0])
        image = os.path.join(self.image_base, split_name, '{}.jpg'.format(item['image']))
        if self.draw_box:
            image = draw_all_box_colored(Image.open(image), item['object'])
        return {'id': item['image'], 'image': image, 'box_info':allbox2str(item['object'], colored=self.draw_box)}


class LvisSource:
    def __init__(self, path='/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/LVIS/lvis_v1_train_for_generation.json', image_base='/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/LVIS/train2017', draw_box=False):
        self.meta = json.load(open(path))
        self.image_base = image_base
        self.draw_box = draw_box
    
    def __len__(self,):
        return len(self.meta)

    def __getitem__(self, index):
        item = self.meta[index]
        # split_name = 'train_{}'.format(item['image'][0])
        # image = os.path.join(self.image_base, split_name, '{}.jpg'.format(item['image']))
        image = os.path.join(self.image_base, '{:012d}.jpg'.format(int(item['image'])))
        if self.draw_box:
            image = draw_all_box_colored(Image.open(image), item['object'])
        return {'id': item['image'], 'image': image, 'box_info':allbox2str(item['object'], colored=self.draw_box)}


if __name__ == "__main__":
    d = OpenImagesSource()
    print(d[3])

