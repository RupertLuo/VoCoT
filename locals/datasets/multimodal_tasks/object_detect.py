import os
import random
import copy
import json
import math
from pathlib import Path
from tkinter import E
from typing import Any

import numpy as np
import torch, tqdm
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

from locals.datasets.image_edit.seg.refcoco import REFER
from locals.datasets.image_edit.seg.grefcoco import G_REFER
from constants import * # ALL_IMG_TOKENS_STR, DEFAULT_BOP_TOKEN, DEFAULT_EOP_TOKEN, DEFAULT_GRD_TOKEN
from ..utils.box_utils import box2str, reshape_box
from collections import defaultdict


def get_num(v):
    return sum([len(c) for c in v.values()])

class LVISDataset(Dataset):
    def __init__(
        self,
        path: str,
        image_dir: str="refcoco",
        avoid_image_gen: bool = False,
        phrase_format: str='special_tokens',
        phrase_prec: int=2,
        phrase_space: bool=False,
        ignore_object: bool=False,
        expand2square: bool=False,
        object_format: str='image',
        specific_object_prop: float=0.5,
        sample_weight: float=1.0,
        min_objects: int=0,
    ):
        self.path = path
        base_name = os.path.splitext(path)[0]
        if os.path.exists(base_name + '_processed.json'):
            print('found processed meta file')
            self.image2anno = json.load(open(base_name + '_processed.json'))
        else:
            print('first time running, processing the raw file')
            tmp_meta = json.load(open(path))
            self.id2category = {item['id']: item for item in tmp_meta['categories']}
            self.image2anno = defaultdict(dict)
            for item in tqdm.tqdm(tmp_meta['annotations'], desc='processing!'):
                class_name = self.id2category[item['category_id']]['name'].replace('_', ' ')
                object_info = {'bbox': item['bbox'], 'class': class_name}
                if class_name not in self.image2anno[item['image_id']]:
                    self.image2anno[item['image_id']][class_name] = []
                self.image2anno[item['image_id']][class_name].append(object_info)
            with open(base_name + '_processed.json', 'w') as wf:
                json.dump(self.image2anno, wf)
        self.image2anno = {k:v for k,v in self.image2anno.items() if get_num(v)>min_objects}
        self.image_keys = list(self.image2anno.keys())
        self.avoid_image_gen = avoid_image_gen
        self.specific_object_prop = specific_object_prop
        self.image_dir = image_dir
        self.phrase_format = phrase_format
        self.phrase_prec = phrase_prec
        self.ignore_object = ignore_object
        self.phrase_space = phrase_space
        self.expand2squre = expand2square
        self.object_format = object_format
        self.sample_weight = sample_weight

        self.detect_prompts = [
            'Please find all key objects in the image.',
            'Detect and locate main objects in the image.'
        ]
        self.objcts_prompts = [
            'Please find {} in the image.',
            'In the image, can you locate the {}?'
        ]
        self.count_prompts = [
            'How many {} are in the image?',
            'How many {} are visible?',
            'What is the number of {} in the image?'
        ]
        print('LVIS-Detection has {} samples'.format(len(self)))

    def __len__(self) -> int:
        return int(len(self.image_keys) * self.sample_weight)
    
    def process_phrase(self, tmp_box, image):
        # crop the image
        width, height = image.size
        box = [c for c in tmp_box]
        box[2] = box[0] + box[2]
        box[3] = box[1] + box[3] 
        x_min, y_min, x_max, y_max = box
        sub_image = image.crop((x_min, y_min, x_max, y_max))
        tmp_box = [c / width if (i % 2==0) else c / height for i,c in enumerate(box)]
        # coor = [round(c*(LOC_TOKEN_NUM-1)) for c in tmp_box]
        # coor = ''.join([ALL_LOC_TOKENS[i] for i in coor])
        if self.expand2squre:
            tmp_box = reshape_box(image, tmp_box)
        coor = box2str(tmp_box, self.phrase_format, self.phrase_prec, space=self.phrase_space)
        coor_format = DEFAULT_BOC_TOKEN + coor + DEFAULT_EOC_TOKEN
        # <p>phrase</p><coor>[LOC]</coor><Img><ImageHere></Img>
        return sub_image, coor_format if self.ignore_object else coor_format + ALL_IMG_TOKENS_STR, tmp_box
    
    def process_group(self, image, object_infos):
        all_boxes = []
        all_boxes_str = []
        for obj in object_infos:
            sub_img, coor_str, box = self.process_phrase(obj['bbox'], image)
            all_boxes.append(box)
            all_boxes_str.append(coor_str)
        return all_boxes, DEFAULT_SEP_TOKEN.join(all_boxes_str)
    
    def __getitem__(self, i: int) -> dict[str, Any]:

        if self.sample_weight >= 1:
            i = i % len(self.image_keys)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)
        
        image_key = self.image_keys[i]
        item = self.image2anno[image_key]
        image_path = os.path.join(self.image_dir, '{:012d}.jpg'.format(int(image_key)))

        image = Image.open(image_path).convert("RGB")

        prob = random.random()
        all_boxes = [[0.0, 0.0, 1.0, 1.0]]
        if prob > self.specific_object_prop:
            task = 'all_objects'
            prompt = random.choice(self.detect_prompts)
            response = 'In the image'
            for obj_name, object_info in item.items():
                # iterate through all objetcs
                obj_all_boxes, obj_all_str = self.process_group(image, object_info)
                response = response + ', find the {}{}'.format(obj_name, obj_all_str)
                all_boxes.extend(obj_all_boxes)
            response = response + '.'
        elif random.random()>0.5:
            task = 'single_object'
            object_name = random.choice(list(item.keys()))
            prompt = random.choice(self.objcts_prompts).format(object_name)
            object_info = item[object_name]
            obj_all_boxes, obj_all_str = self.process_group(image, object_info)
            response = 'Find {}, get {}{}.'.format(object_name, object_name, obj_all_str)
            num_objects = len(obj_all_boxes)
            all_boxes.extend(obj_all_boxes)
        else:
            task = 'count_object'
            object_name = random.choice(list(item.keys()))
            prompt = random.choice(self.count_prompts).format(object_name) + ' ' + COT_ACTIVATION
            object_info = item[object_name]
            obj_all_boxes, obj_all_str = self.process_group(image, object_info)
            num_objects = len(obj_all_boxes)
            all_boxes.extend(obj_all_boxes)
            response = 'Find {}, get {}{}. So the answer is {}.'.format(object_name, object_name, obj_all_str, num_objects)
        
        input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + prompt
        
        # image_1 = Image.fromarray(image_1)
        sources = [{'from': 'human', 'value': input_sent},
                   {'from': 'gpt', 'value': response}]

        if self.object_format == 'image':
            raise ValueError
            # return {'input_images': [image, sub_image] if not self.ignore_object else [image], 'conversation': sources, 'image_label_masks': image_label_masks}
        elif self.object_format == 'representation':
            return {'input_images': [image] , 'conversation': sources, 'image_label_masks': [0]*len(all_boxes), 'box': all_boxes}
        else:
            raise ValueError