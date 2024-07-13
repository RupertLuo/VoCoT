# --------------------------------------------------------
# InstructDiffusion
# Based on instruct-pix2pix (https://github.com/timothybrooks/instruct-pix2pix)
# Modified by Binxin Yang (tennyson@mail.ustc.edu.cn)
# --------------------------------------------------------

from __future__ import annotations
from email.policy import default

import os
import random
import copy
import json
import math
from pathlib import Path
from tkinter import E
from typing import Any

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

from locals.datasets.image_edit.seg.refcoco import REFER
from locals.datasets.image_edit.seg.grefcoco import G_REFER
from constants import * # ALL_IMG_TOKENS_STR, DEFAULT_BOP_TOKEN, DEFAULT_EOP_TOKEN, DEFAULT_GRD_TOKEN
from ..utils.box_utils import box2str, reshape_box, resize_image_to_square
from collections import defaultdict
from ..utils.box_utils import *

class RefCOCODataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        dataset_name: str="refcoco",
        split_by: str='unc',
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        transparency: float = 0.0,
        test: bool = False,
        image_path: str = None,
        task_mode: str = 't2i',
        avoid_image_gen: bool = False,
        phrase_format: str='special_tokens',
        phrase_prec: int=2,
        phrase_space: bool=False,
        ignore_object: bool=False,
        expand2square: bool=False,
        object_format: str='image'
    ):
        assert split in ("train", "val", "test", "testA", "testB")
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.avoid_image_gen = avoid_image_gen
        self.phrase_format = phrase_format
        self.phrase_prec = phrase_prec
        self.ignore_object = ignore_object
        self.object_format = object_format
        if self.object_format in ['text', 'coordinate']:
            self.ignore_object = True
        self.phrase_space = phrase_space
        self.G_ref_dataset=REFER(data_root=path, dataset=dataset_name, splitBy=split_by)
        self.IMAGE_DIR = os.path.join(image_path, 'train2014')
        self.list_ref=self.G_ref_dataset.getRefIds(split=split)
        self.transparency = transparency
        self.test = test
        self.task_mode = task_mode
        self.expand2squre = expand2square

        seg_diverse_prompt_path_i2t = 'locals/datasets/prompts/prompt_ref_i2t.txt'
        seg_diverse_prompt_path_t2i = 'locals/datasets/prompts/prompt_ref_t2i.txt'
        self.seg_diverse_prompt_list_i2t=[]
        with open(seg_diverse_prompt_path_i2t) as f:
            line=f.readline()
            while line:
                line=line.strip('\n')
                self.seg_diverse_prompt_list_i2t.append(line)
                line=f.readline()
        
        self.seg_diverse_prompt_list_t2i=[]
        with open(seg_diverse_prompt_path_t2i) as f:
            line=f.readline()
            while line:
                line=line.strip('\n')
                self.seg_diverse_prompt_list_t2i.append(line)
                line=f.readline()

    def __len__(self) -> int:
        return len(self.list_ref)
    
    def process_phrase(self, box, image):
        # crop the image
        width, height = image.size
        x_min, y_min, x_max, y_max = box
        sub_image = image.crop((x_min, y_min, x_max, y_max))
        sub_image = resize_image_to_square(sub_image)
        tmp_box = [c / width if (i % 2==0) else c / height for i,c in enumerate(box)]
        # coor = [round(c*(LOC_TOKEN_NUM-1)) for c in tmp_box]
        # coor = ''.join([ALL_LOC_TOKENS[i] for i in coor])
        if self.expand2squre:
            tmp_box = reshape_box(image, tmp_box)
        coor = box2str(tmp_box, self.phrase_format, self.phrase_prec, space=self.phrase_space)
        coor_format = DEFAULT_BOC_TOKEN + coor + DEFAULT_EOC_TOKEN
        if self.object_format == 'text':
            coor_format = coor_format.replace(DEFAULT_BOC_TOKEN, '[').replace(DEFAULT_EOC_TOKEN, ']')
        # <p>phrase</p><coor>[LOC]</coor><Img><ImageHere></Img>
        return sub_image, coor_format if self.ignore_object else coor_format + ALL_IMG_TOKENS_STR, tmp_box
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        
        ref_ids = self.list_ref[i] 
        ref = self.G_ref_dataset.loadRefs(ref_ids)[0]
        phrase = random.choice(ref['sentences'])['sent']

        image_name = self.G_ref_dataset.loadImgs(ref['image_id'])[0]['file_name']
        image_path = os.path.join(self.IMAGE_DIR,image_name)

        image = Image.open(image_path).convert("RGB")

        if self.task_mode == 'both':
            prob = random.random()
            if prob > 0.5:
                task = 'i2t'
            else:
                task = 't2i'
        else:
            task = self.task_mode
        
        # phrase and bounding box info
        phrase = DEFAULT_BOP_TOKEN + phrase + DEFAULT_EOP_TOKEN
        region_info = copy.deepcopy(self.G_ref_dataset.getRefBox(ref_ids))
        region_info[2] = region_info[2] + region_info[0]
        region_info[3] = region_info[3] + region_info[1]
        sub_image, region_info, box_info = self.process_phrase(region_info, image)
        all_box_info = [[0.0, 0.0, 1.0, 1.0], box_info]
        if task == 'i2t':
            prompt = random.choice(self.seg_diverse_prompt_list_i2t)
            input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + ' ' + prompt.format(region=region_info)
            output_sent = phrase
            if self.ignore_object:
                image_label_masks = [0]
            else:
                image_label_masks = [0, 0]
        else:
            prompt = random.choice(self.seg_diverse_prompt_list_t2i)
            input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + ' ' + prompt.format(phrase=phrase)
            output_sent = region_info
            if self.avoid_image_gen:
                if self.ignore_object:
                    image_label_masks = [0]
                else:
                    image_label_masks = [0, 0]
            else:
                if self.ignore_object:
                    image_label_masks = [0]
                else:
                    image_label_masks = [0, 1]
        
        # image_1 = Image.fromarray(image_1)
        if self.object_format == 'text':
            # remove the <grd> token when mearely using text as CoT
            input_sent = input_sent.replace(DEFAULT_GRD_TOKEN, '')
        sources = [{'from': 'human', 'value': input_sent},
                   {'from': 'gpt', 'value': output_sent}]

        if self.object_format == 'image':
            return {'input_images': [image, sub_image] if not self.ignore_object else [image], 'conversation': sources, 'image_label_masks': image_label_masks}
        elif self.object_format == 'representation':
            return {'input_images': [image] , 'conversation': sources, 'image_label_masks': image_label_masks, 'box': all_box_info}
        elif self.object_format == 'text':
            return {'input_images': [image] , 'conversation': sources, 'image_label_masks': image_label_masks, 'box': all_box_info[:1]}
        elif self.object_format == 'coordinate':
            return {'input_images': [image], 'conversation': sources, 'image_label_masks': image_label_masks, 'box': all_box_info[:1]}
        else:
            raise ValueError

class RefCOCOEvalDataset(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        dataset_name: str="refcoco",
        split_by: str='unc',
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        transparency: float = 0.0,
        test: bool = False,
        image_path: str = None,
        task_mode: str = 't2i',
        phrase_format: str='special_tokens',
        phrase_prec: int=2,
        phrase_space: bool=False,
        naive_format: bool = False,
        for_llava: bool=False,
        expand2square: bool=False,
        object_format: str='image',
    ):
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.phrase_space = phrase_space
        self.G_ref_dataset=REFER(data_root=path, dataset=dataset_name, splitBy=split_by)
        self.IMAGE_DIR = os.path.join(image_path, 'train2014')
        self.list_ref=self.G_ref_dataset.getRefIds(split=split)
        self.naive_format = naive_format
        self.for_llava = for_llava
        self.expand2square = expand2square
        self.object_format = object_format
        # processing all descriptions
        self.meta = []
        for ref_ids in self.list_ref:
            ref = self.G_ref_dataset.loadRefs(ref_ids)[0]
            image_name = self.G_ref_dataset.loadImgs(ref['image_id'])[0]['file_name']
            region_info = copy.deepcopy(self.G_ref_dataset.getRefBox(ref_ids))
            region_info[2] = region_info[2] + region_info[0]
            region_info[3] = region_info[3] + region_info[1]
            for phrase in ref['sentences']:
                self.meta.append({
                    'image': image_name,
                    'phrase': phrase['sent'],
                    'label': region_info
                })
        self.transparency = transparency
        self.test = test
        self.task_mode = task_mode
        self.phrase_format = phrase_format
        self.phrase_prec = phrase_prec

        seg_diverse_prompt_path_i2t = 'locals/datasets/prompts/prompt_ref_i2t.txt'
        seg_diverse_prompt_path_t2i = 'locals/datasets/prompts/prompt_ref_t2i.txt'
        self.seg_diverse_prompt_list_i2t=[]
        with open(seg_diverse_prompt_path_i2t) as f:
            line=f.readline()
            while line:
                line=line.strip('\n')
                self.seg_diverse_prompt_list_i2t.append(line)
                line=f.readline()
        
        self.seg_diverse_prompt_list_t2i=[]
        with open(seg_diverse_prompt_path_t2i) as f:
            line=f.readline()
            while line:
                line=line.strip('\n')
                self.seg_diverse_prompt_list_t2i.append(line)
                line=f.readline()

    def __len__(self) -> int:
        return len(self.meta)
    
    def norm_box(self, box, image):
        # crop the image
        width, height = image.size
        tmp_box = [c / width if (i % 2==0) else c / height for i,c in enumerate(box)]
        if self.expand2square:
            tmp_box = reshape_box(image, tmp_box)
        return tmp_box
    
    def process_phrase(self, box, image):
        # crop the image
        width, height = image.size
        x_min, y_min, x_max, y_max = box
        sub_image = image.crop((x_min, y_min, x_max, y_max))
        tmp_box = [c / width if (i % 2==0) else c / height for i,c in enumerate(box)]
        # coor = [round(c*(LOC_TOKEN_NUM-1)) for c in tmp_box]
        # coor = ''.join([ALL_LOC_TOKENS[i] for i in coor])
        if self.expand2square:
            tmp_box = reshape_box(image, tmp_box)
        coor = box2str(tmp_box, self.phrase_format, self.phrase_prec, self.phrase_space)
        coor_format = DEFAULT_BOC_TOKEN + coor + DEFAULT_EOC_TOKEN
        # <p>phrase</p><coor>[LOC]</coor><Img><ImageHere></Img>
        return sub_image, coor_format + ALL_IMG_TOKENS_STR, tmp_box
    
    def getlabel(self, i):
        item = self.meta[i] 
        phrase = item['phrase']

        image_name = item['image']
        image_path = os.path.join(self.IMAGE_DIR,image_name)

        image = Image.open(image_path).convert("RGB")

        if self.task_mode == 'both':
            prob = random.random()
            if prob > 0.5:
                task = 'i2t'
            else:
                task = 't2i'
        else:
            task = self.task_mode
        
        # phrase and bounding box info
        phrase = DEFAULT_BOP_TOKEN + phrase + DEFAULT_EOP_TOKEN
        region_info = item['label']
        region_info = self.norm_box(region_info, image)
        if task == 'i2t':
            return phrase
        else:
            return region_info
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        
        item = self.meta[i]
        phrase = item['phrase']

        image_name = item['image']
        image_path = os.path.join(self.IMAGE_DIR,image_name)

        image = Image.open(image_path).convert("RGB")

        if self.task_mode == 'both':
            prob = random.random()
            if prob > 0.5:
                task = 'i2t'
            else:
                task = 't2i'
        else:
            task = self.task_mode
        
        # phrase and bounding box info
        # if self.naive_format:
        #     return {'input_images': image, 'phrase': phrase, 'item_id': 'refcoco_{}'.format(i)}
        if not self.naive_format:
            phrase = DEFAULT_BOP_TOKEN + phrase + DEFAULT_EOP_TOKEN
            
        region_info = item['label']
        sub_image, region_info, box_info = self.process_phrase(region_info, image)
        
        all_box_info = [[0.0, 0.0, 1.0, 1.0]]
        if task == 'i2t':
            prompt = random.choice(self.seg_diverse_prompt_list_i2t)
            if self.naive_format:
                input_sent = ' Please provide a short description for this region: ' + region_info
            else:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + ' ' + prompt.format(region=region_info)
            output_sent = phrase
            image_label_masks = [0, 0]
            all_box_info.append(box_info)
        else:
            # prompt = random.choice(self.seg_diverse_prompt_list_t2i)
            prompt = self.seg_diverse_prompt_list_t2i[2]
            if self.naive_format:
                input_sent = ' Please provide the bounding box coordinate of the region this sentence describes: {}.'.format(phrase)
            else:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + ' ' + prompt.format(phrase=phrase)
            output_sent = region_info
            image_label_masks = [0, 0]
        
        # image_1 = Image.fromarray(image_1)
        if self.naive_format:
            return {'input_images': [image], 'question': input_sent, 'item_id': 'eval_{}'.format(i)}
        sources = [{'from': 'human', 'value': input_sent}]
        if self.object_format == 'image':
            return {'input_images': [image, sub_image], 'conversation': sources, 'image_label_masks': image_label_masks}
        elif self.object_format == 'representation':
            return {'input_images': [image], 'conversation': sources, 'box': all_box_info}
        else:
            raise NotImplementedError


class GRefCOCODataset(RefCOCODataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        dataset_name: str="refcoco",
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        transparency: float = 0.0,
        test: bool = False,
        image_path: str = None,
        task_mode: str = 't2i',
        avoid_image_gen: bool = False,
        ignore_object: bool = False,
        phrase_format: str='special_tokens',
        phrase_prec: int=2,
        phrase_space: bool=False,
        expand2square: bool=False,
        object_format: str='image',
    ):
        assert split in ("train", "val", "test")
        self.path = path
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.avoid_image_gen = avoid_image_gen
        self.ignore_object = ignore_object
        self.object_format = object_format
        if self.object_format in ['text', 'coordinate']:
            self.ignore_object = True
        self.G_ref_dataset=G_REFER(data_root=path)
        self.IMAGE_DIR = os.path.join(image_path, 'train2014')
        self.list_ref=self.G_ref_dataset.getRefIds(split=split)
        self.transparency = transparency
        self.test = test
        self.task_mode = task_mode
        self.phrase_format = phrase_format
        self.phrase_prec = phrase_prec
        self.phrase_space = phrase_space
        self.expand2squre = expand2square

        seg_diverse_prompt_path_i2t = 'locals/datasets/prompts/prompt_ref_i2t.txt'
        seg_diverse_prompt_path_t2i = 'locals/datasets/prompts/prompt_ref_t2i.txt'
        self.seg_diverse_prompt_list_i2t=[]
        with open(seg_diverse_prompt_path_i2t) as f:
            line=f.readline()
            while line:
                line=line.strip('\n')
                self.seg_diverse_prompt_list_i2t.append(line)
                line=f.readline()
        
        self.seg_diverse_prompt_list_t2i=[]
        with open(seg_diverse_prompt_path_t2i) as f:
            line=f.readline()
            while line:
                line=line.strip('\n')
                self.seg_diverse_prompt_list_t2i.append(line)
                line=f.readline()

    def __getitem__(self, i: int) -> dict[str, Any]:
    
        ref_ids = self.list_ref[i] 
        ref = self.G_ref_dataset.loadRefs(ref_ids)[0]
        phrase = random.choice(ref['sentences'])['sent']

        image_name = self.G_ref_dataset.loadImgs(ref['image_id'])[0]['file_name']
        image_path = os.path.join(self.IMAGE_DIR,image_name)

        image = Image.open(image_path).convert("RGB")

        if self.task_mode == 'both':
            prob = random.random()
            if prob > 0.5:
                task = 'i2t'
            else:
                task = 't2i'
        else:
            task = self.task_mode
        
        # phrase and bounding box info
        phrase = DEFAULT_BOP_TOKEN + phrase + DEFAULT_EOP_TOKEN
        region_info = copy.deepcopy(self.G_ref_dataset.getRefBox(ref_ids))
        if len(region_info) == 0:
            # no bounding box
            task = 't2i'
            region_info = 'No regions found.'
            sub_images = []
            all_box_info = [[0.0, 0.0, 1.0, 1.0]]
        else:
            sub_images = []
            reg_texts = []
            all_box_info = [[0.0, 0.0, 1.0, 1.0]]
            for reg in region_info:
                reg[2] = reg[2] + reg[0]
                reg[3] = reg[3] + reg[1]
                sub_image, reg_txt, tmp_box = self.process_phrase(reg, image)
                all_box_info.append(tmp_box)
                sub_images.append(sub_image)
                reg_texts.append(reg_txt)
            region_info = DEFAULT_SEP_TOKEN.join(reg_texts)
        
        if task == 'i2t':
            prompt = random.choice(self.seg_diverse_prompt_list_i2t)
            input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + ' ' + prompt.format(region=region_info)
            output_sent = phrase
            if self.ignore_object:
                image_label_masks = [0]
            else:
                image_label_masks = [0] + [0]*len(sub_images)
        else:
            prompt = random.choice(self.seg_diverse_prompt_list_t2i)
            input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + ' ' + prompt.format(phrase=phrase)
            output_sent = region_info
            if self.ignore_object:
                image_label_masks = [0]
            elif self.avoid_image_gen:
                image_label_masks = [0] + [0]*len(sub_images)
            else:
                image_label_masks = [0] + [1]*len(sub_images)
        
        # image_1 = Image.fromarray(image_1)
        if self.object_format == 'text':
            input_sent = input_sent.replace(DEFAULT_GRD_TOKEN, '')
        sources = [{'from': 'human', 'value': input_sent},
                {'from': 'gpt', 'value': output_sent}]

        if self.object_format == 'image':
            return {'input_images': [image] + sub_images if not self.ignore_object else [image], 'conversation': sources, 'image_label_masks': image_label_masks}
        elif self.object_format == 'representation':
            return {'input_images': [image], 'conversation': sources, 'image_label_masks': image_label_masks, 'box': all_box_info}
        elif self.object_format == 'text':
            return {'input_images': [image] , 'conversation': sources, 'image_label_masks': image_label_masks, 'box': all_box_info[:1]}
        elif self.object_format == 'coordinate':
            return {'input_images': [image] , 'conversation': sources, 'image_label_masks': image_label_masks, 'box': all_box_info[:1]}
        else:
            raise ValueError

class VGGroundDataset(RefCOCODataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        dataset_name: str="refcoco",
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        transparency: float = 0.0,
        test: bool = False,
        image_path: str = None,
        task_mode: str = 't2i',
        avoid_image_gen: bool = False,
        ignore_object: bool = False,
        phrase_format: str='special_tokens',
        phrase_prec: int=2,
        max_objects: int= 10,
    ):
        assert split in ("train", "val", "test")
        self.path = path
        # loading the data and merge
        self.meta = defaultdict(list)
        with open(path, 'r') as rf:
            print('loading and merging VG')
            for line in rf:
                info = json.loads(line.strip())
                self.meta[info['img_path']].append(info)
        self.meta_info = []
        for k,v in self.meta.items():
            item = {'image_path': k, 'height': v[0]['height'], 'width': v[0]['width'], 'expressions':[]}
            for obj in v:
                item['expressions'].append({
                    'phrase': obj['expression'], 'bbox': obj['bbox']
                })
            self.meta_info.append(item)
        del(self.meta)

        self.avoid_image_gen = avoid_image_gen
        self.ignore_object = ignore_object
        self.IMAGE_DIR = image_path
        self.task_mode = task_mode
        self.phrase_format = phrase_format
        self.phrase_prec = phrase_prec
        self.max_objects = max_objects

        seg_diverse_prompt_path_i2t = 'locals/datasets/prompts/prompt_ref_i2t.txt'
        seg_diverse_prompt_path_t2i = 'locals/datasets/prompts/prompt_ref_t2i.txt'
        self.seg_diverse_prompt_list_i2t=[]
        with open(seg_diverse_prompt_path_i2t) as f:
            line=f.readline()
            while line:
                line=line.strip('\n')
                self.seg_diverse_prompt_list_i2t.append(line)
                line=f.readline()
        
        self.seg_diverse_prompt_list_t2i=[]
        with open(seg_diverse_prompt_path_t2i) as f:
            line=f.readline()
            while line:
                line=line.strip('\n')
                self.seg_diverse_prompt_list_t2i.append(line)
                line=f.readline()

    def __len__(self):
        return len(self.meta_info)
    
    def __getitem__(self, index: int) -> dict[str, Any]:
    
        item = self.meta_info[index] 

        image_name = item['image_path']
        image_name = image_name[7:] if image_name.startswith('images/') else image_name
        image_name = image_name[8:] if image_name.startswith('images2/') else image_name
        image_path = os.path.join(self.IMAGE_DIR,image_name)

        image = Image.open(image_path).convert("RGB")

        convs = []
        sub_images = []
        image_label_masks = [0]
        valid_exps = [i for i in item['expressions']]
        random.shuffle(valid_exps)
        for i, exp in enumerate(valid_exps[:self.max_objects]):

            if self.task_mode == 'both':
                prob = random.random()
                if prob > 0.5:
                    task = 'i2t'
                else:
                    task = 't2i'
            else:
                task = self.task_mode
        
            # phrase and bounding box info
            phrase = exp['phrase']
            phrase = DEFAULT_BOP_TOKEN + phrase + DEFAULT_EOP_TOKEN
            region_info = copy.deepcopy(exp['bbox'])


            sub_image, reg_txt = self.process_phrase(region_info, image)
            sub_images.append(sub_image)
            region_info = reg_txt
        
            input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + ' ' if i == 0 else ''
            if task == 'i2t':
                prompt = random.choice(self.seg_diverse_prompt_list_i2t)
                input_sent = input_sent + prompt.format(region=region_info)
                output_sent = phrase
                if self.ignore_object:
                    image_label_masks.extend([])
                else:
                    image_label_masks.extend([0])
            else:
                prompt = random.choice(self.seg_diverse_prompt_list_t2i)
                input_sent = input_sent + prompt.format(phrase=phrase)
                output_sent = region_info
                if self.ignore_object:
                    image_label_masks.extend([])
                elif self.avoid_image_gen:
                    image_label_masks.extend([0])
                else:
                    image_label_masks.extend([1])
            convs.extend([{'from': 'human', 'value': input_sent},
                {'from': 'gpt', 'value': output_sent}])

        return {'input_images': [image] + sub_images if not self.ignore_object else [image], 'conversation': convs, 'image_label_masks': image_label_masks}



class ClevrRefDataset(Dataset):
    def __init__(
        self,
        path: str,
        image_path: str = None,
        phrase_format: str='special_tokens',
        phrase_prec: int=2,
        phrase_space: bool=False,
        naive_format: bool = False,
        for_llava: bool=False,
        expand2square: bool=False,
        object_format: str='image',
        require_cot: bool=False
    ):
        self.path = path
        self.phrase_space = phrase_space
        self.meta = json.load(open(path))
        self.IMAGE_DIR = image_path
        self.naive_format = naive_format
        self.naive_format = for_llava
        self.expand2square = expand2square
        self.object_format = object_format
        self.require_cot = require_cot

        self.phrase_format = phrase_format
        self.phrase_prec = phrase_prec
        self.use_image_path = False
        self.minigptv2 = False

        seg_diverse_prompt_path_i2t = 'locals/datasets/prompts/prompt_ref_i2t.txt'
        seg_diverse_prompt_path_t2i = 'locals/datasets/prompts/prompt_ref_t2i.txt'
        self.seg_diverse_prompt_list_i2t=[]
        with open(seg_diverse_prompt_path_i2t) as f:
            line=f.readline()
            while line:
                line=line.strip('\n')
                self.seg_diverse_prompt_list_i2t.append(line)
                line=f.readline()
        
        self.seg_diverse_prompt_list_t2i=[]
        with open(seg_diverse_prompt_path_t2i) as f:
            line=f.readline()
            while line:
                line=line.strip('\n')
                self.seg_diverse_prompt_list_t2i.append(line)
                line=f.readline()

    def __len__(self) -> int:
        return len(self.meta)
    
    def norm_box(self, box, image):
        # crop the image
        width, height = image.size
        tmp_box = [c / width if (i % 2==0) else c / height for i,c in enumerate(box)]
        if self.expand2square:
            tmp_box = reshape_box(image, tmp_box)
        return tmp_box
    
    def process_phrase(self, box, image):
        # crop the image
        width, height = image.size
        x_min, y_min, x_max, y_max = box
        sub_image = image.crop((x_min, y_min, x_max, y_max))
        tmp_box = [c / width if (i % 2==0) else c / height for i,c in enumerate(box)]
        # coor = [round(c*(LOC_TOKEN_NUM-1)) for c in tmp_box]
        # coor = ''.join([ALL_LOC_TOKENS[i] for i in coor])
        if self.expand2square:
            tmp_box = reshape_box(image, tmp_box)
        coor = box2str(tmp_box, self.phrase_format, self.phrase_prec, self.phrase_space)
        coor_format = DEFAULT_BOC_TOKEN + coor + DEFAULT_EOC_TOKEN
        # <p>phrase</p><coor>[LOC]</coor><Img><ImageHere></Img>
        return sub_image, coor_format + ALL_IMG_TOKENS_STR, tmp_box
    
    def getlabel(self, i):
        item = self.meta[i] 
        phrase = item['text']

        image_name = item['image']
        image_path = os.path.join(self.IMAGE_DIR,image_name)

        image = Image.open(image_path).convert("RGB")
        region_info = item['bbox']
        region_info = self.norm_box(region_info, image)
        return region_info
    
    def proc_phrase(self, txt):
        if txt.startswith('Find '):
            txt = txt[5:]
        return txt[0].lower() + txt[1:]

    def cot_turn(self, i, thought=None, thought_ids=None, eoc_id=None, img_id=None, mistral=False):
        input_dict = self[i]
        all_box = [[0.0, 0.0, 1.0, 1.0]]
        new_thought, thought_boxes = process_thought(thought, mistral=mistral)
        all_box.extend(thought_boxes)
        input_dict['conversation'].append(
            {'from': 'gpt', 'value': new_thought}
        )
        input_dict['conversation'].append(
            {'from': 'human', 'value': 'What is your final answer?'}
        )
        input_dict['box'] = all_box
        if thought_ids is not None:
            thought_ids = thought_ids.squeeze()
            if mistral:
                suffix = torch.tensor([   733, 16289, 28793,  1824,   349,   574,  1480,  4372, 28804,   733,
                                        28748, 16289, 28793]).to(thought_ids.device)
            else:
                suffix = torch.tensor([ 3148,  1001, 29901,  1724,   338,   596,  2186,  1234, 29973,   319,
                     1799,  9047, 13566, 29901]).to(thought_ids.device)
            del(input_dict['conversation'])
            eoc_indices = [-1] + torch.where(thought_ids == eoc_id)[0].tolist() + [thought_ids.shape[0]-1]
            input_dict['input_ids'] = []
            for i in range(len(eoc_indices) - 1):
                input_dict['input_ids'].append(thought_ids[eoc_indices[i]+1:eoc_indices[i+1]+1])
                if i < len(eoc_indices) - 2:
                    if thought_ids[eoc_indices[i+1]+1].item() != img_id:
                        input_dict['input_ids'].append(torch.tensor([img_id]).to(thought_ids.device))
            input_dict['input_ids'].append(suffix)
            input_dict['input_ids'] = torch.cat(input_dict['input_ids'])
        return input_dict
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        
        item = self.meta[i]
        phrase = self.proc_phrase(item['text'])

        image_name = item['image']
        image_path = os.path.join(self.IMAGE_DIR,image_name)

        image = Image.open(image_path).convert("RGB")

        
        # phrase and bounding box info
        # if self.naive_format:
        #     return {'image': image, 'phrase': phrase, 'item_id': 'refcoco_{}'.format(i)}
        if not self.naive_format:
            phrase = DEFAULT_BOP_TOKEN + phrase + DEFAULT_EOP_TOKEN
            
        region_info = item['bbox']
        sub_image, region_info, box_info = self.process_phrase(region_info, image)
        
        all_box_info = [[0.0, 0.0, 1.0, 1.0]]
        prompt = self.seg_diverse_prompt_list_t2i[2]
        if self.naive_format:
            if self.minigptv2:
                input_sent = '[refer] give me the location of {}.'.format(phrase)
            else:
                input_sent =  'Please provide the bounding box coordinate of the region this sentence describes: {}.'.format(phrase)
        else:
            if self.require_cot:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + prompt.format(phrase=phrase) + COT_ACTIVATION
            else:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + prompt.format(phrase=phrase)
        output_sent = region_info
        image_label_masks = [0, 0]
        
        # image_1 = Image.fromarray(image_1)
        if self.use_image_path:
            image = image_path
        if self.naive_format:
            return {'input_images': [image], 'question': input_sent, 'item_id': 'eval_{}'.format(i)}
        sources = [{'from': 'human', 'value': input_sent}]
        if self.object_format == 'image':
            return {'input_images': [image, sub_image], 'conversation': sources, 'image_label_masks': image_label_masks}
        elif self.object_format == 'representation':
            return {'input_images': [image], 'conversation': sources, 'box': all_box_info}
        else:
            raise NotImplementedError


class GuessWhatDataset(Dataset):
    def __init__(
        self,
        path: str,
        image_path: str = None,
        phrase_format: str='special_tokens',
        phrase_prec: int=2,
        phrase_space: bool=False,
        naive_format: bool = False,
        for_llava: bool=False,
        expand2square: bool=False,
        object_format: str='image',
        require_cot: bool=False
    ):
        self.path = path
        self.phrase_space = phrase_space
        self.meta = [json.load(line) for line in open(path)]
        self.IMAGE_DIR = image_path
        self.naive_format = naive_format
        self.for_llava = for_llava
        self.expand2square = expand2square
        self.object_format = object_format
        self.require_cot = require_cot

        self.phrase_format = phrase_format
        self.phrase_prec = phrase_prec

        seg_diverse_prompt_path_i2t = 'locals/datasets/prompts/prompt_ref_i2t.txt'
        seg_diverse_prompt_path_t2i = 'locals/datasets/prompts/prompt_ref_t2i.txt'
        self.seg_diverse_prompt_list_i2t=[]
        with open(seg_diverse_prompt_path_i2t) as f:
            line=f.readline()
            while line:
                line=line.strip('\n')
                self.seg_diverse_prompt_list_i2t.append(line)
                line=f.readline()
        
        self.seg_diverse_prompt_list_t2i=[]
        with open(seg_diverse_prompt_path_t2i) as f:
            line=f.readline()
            while line:
                line=line.strip('\n')
                self.seg_diverse_prompt_list_t2i.append(line)
                line=f.readline()

    def __len__(self) -> int:
        return len(self.meta)
    
    def norm_box(self, box, image):
        # crop the image
        width, height = image.size
        tmp_box = [c / width if (i % 2==0) else c / height for i,c in enumerate(box)]
        if self.expand2square:
            tmp_box = reshape_box(image, tmp_box)
        return tmp_box
    
    def process_phrase(self, box, image):
        # crop the image
        width, height = image.size
        x_min, y_min, x_max, y_max = box
        sub_image = image.crop((x_min, y_min, x_max, y_max))
        tmp_box = [c / width if (i % 2==0) else c / height for i,c in enumerate(box)]
        # coor = [round(c*(LOC_TOKEN_NUM-1)) for c in tmp_box]
        # coor = ''.join([ALL_LOC_TOKENS[i] for i in coor])
        if self.expand2square:
            tmp_box = reshape_box(image, tmp_box)
        coor = box2str(tmp_box, self.phrase_format, self.phrase_prec, self.phrase_space)
        coor_format = DEFAULT_BOC_TOKEN + coor + DEFAULT_EOC_TOKEN
        # <p>phrase</p><coor>[LOC]</coor><Img><ImageHere></Img>
        return sub_image, coor_format + ALL_IMG_TOKENS_STR, tmp_box
    
    def getlabel(self, i):
        item = self.meta[i] 
        phrase = item['text']

        image_name = item['image']
        image_path = os.path.join(self.IMAGE_DIR,image_name)

        image = Image.open(image_path).convert("RGB")
        region_info = item['bbox']
        region_info = self.norm_box(region_info, image)
        return region_info
    
    def proc_phrase(self, txt):
        if txt.startswith('Find '):
            txt = txt[5:]
        return txt[0].lower() + txt[1:]

    def cot_turn(self, i, thought=None, thought_ids=None, eoc_id=None, img_id=None, mistral=False):
        input_dict = self[i]
        all_box = [[0.0, 0.0, 1.0, 1.0]]
        new_thought, thought_boxes = process_thought(thought, mistral=mistral)
        all_box.extend(thought_boxes)
        input_dict['conversation'].append(
            {'from': 'gpt', 'value': new_thought}
        )
        input_dict['conversation'].append(
            {'from': 'human', 'value': 'What is your final answer?'}
        )
        input_dict['box'] = all_box
        if thought_ids is not None:
            thought_ids = thought_ids.squeeze()
            if mistral:
                suffix = torch.tensor([   733, 16289, 28793,  1824,   349,   574,  1480,  4372, 28804,   733,
                                        28748, 16289, 28793]).to(thought_ids.device)
            else:
                suffix = torch.tensor([ 3148,  1001, 29901,  1724,   338,   596,  2186,  1234, 29973,   319,
                     1799,  9047, 13566, 29901]).to(thought_ids.device)
            del(input_dict['conversation'])
            eoc_indices = [-1] + torch.where(thought_ids == eoc_id)[0].tolist() + [thought_ids.shape[0]-1]
            input_dict['input_ids'] = []
            for i in range(len(eoc_indices) - 1):
                input_dict['input_ids'].append(thought_ids[eoc_indices[i]+1:eoc_indices[i+1]+1])
                if i < len(eoc_indices) - 2:
                    if thought_ids[eoc_indices[i+1]+1].item() != img_id:
                        input_dict['input_ids'].append(torch.tensor([img_id]).to(thought_ids.device))
            input_dict['input_ids'].append(suffix)
            input_dict['input_ids'] = torch.cat(input_dict['input_ids'])
        return input_dict
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        
        item = self.meta[i]
        phrase = self.proc_phrase(item['text'])

        image_name = item['image']
        image_path = os.path.join(self.IMAGE_DIR,image_name)

        image = Image.open(image_path).convert("RGB")

        
        # phrase and bounding box info
        if self.naive_format:
            return {'image': image, 'phrase': phrase, 'item_id': 'refcoco_{}'.format(i)}
        if not self.for_llava:
            phrase = DEFAULT_BOP_TOKEN + phrase + DEFAULT_EOP_TOKEN
            
        region_info = item['bbox']
        sub_image, region_info, box_info = self.process_phrase(region_info, image)
        
        all_box_info = [[0.0, 0.0, 1.0, 1.0]]
        prompt = self.seg_diverse_prompt_list_t2i[2]
        if self.for_llava:
            input_sent = ALL_IMG_TOKENS_STR + ' Please provide the bounding box coordinate of the region this sentence describes: {}.'.format(phrase)
        else:
            if self.require_cot:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + prompt.format(phrase=phrase) + COT_ACTIVATION
            else:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + prompt.format(phrase=phrase)
        output_sent = region_info
        image_label_masks = [0, 0]
        
        # image_1 = Image.fromarray(image_1)
        sources = [{'from': 'human', 'value': input_sent}]
        if self.object_format == 'image':
            return {'input_images': [image, sub_image], 'conversation': sources, 'image_label_masks': image_label_masks}
        elif self.object_format == 'representation':
            return {'input_images': [image], 'conversation': sources, 'box': all_box_info}
        else:
            raise NotImplementedError