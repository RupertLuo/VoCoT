prompt = "{question}"

from importlib.metadata import requires
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
import pandas as pd
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset
from utils.util import byte2image
from ..utils.box_utils import *
from datasets import load_dataset

from constants import * # ALL_IMG_TOKENS_STR, DEFAULT_BOP_TOKEN, DEFAULT_EOP_TOKEN, DEFAULT_GRD_TOKEN

all_options = ['A', 'B', 'C', 'D']

def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options

class MMVetDataset(Dataset):
    def __init__(
        self,
        path: str,
        base_path: str,
    ):
        self.path = path
        self.meta = json.load(open(path))
        self.base_path = base_path
        self.keys = list(self.meta.keys())

    def __len__(self) -> int:
        return len(self.keys)
    
    def getlabel(self, i):
        key = self.keys[i]
        return self.meta[key]['answer']

    def __getitem__(self, i: int) -> dict[str, Any]:
        
        key = self.keys[i]
        item = self.meta[key]
        question = item['question']
        image_path = os.path.join(self.base_path, item['imagename'])
        input_sent = ALL_IMG_TOKENS_STR + ' ' + prompt.format(question=question)

        image = Image.open(image_path).convert("RGB")
        
        # image_1 = Image.fromarray(image_1)
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources}

class MMBenchDataset(Dataset):
    def __init__(
        self,
        path: str,
        single_pred_prompt: bool=False,
        require_cot: bool=False,
    ):
        self.path = path
        self.meta = pd.read_table(path)
        self.single_pred_prompt = single_pred_prompt
        self.require_cot = require_cot

    def __len__(self) -> int:
        return self.meta.shape[0]
    
    def getlabel(self, i):
        item = self.meta.iloc[i]
        return item['answer']

    def cot_turn(self, i, thought=None, thought_ids=None, eoc_id=None, img_id=None):
        input_dict = self[i]
        all_box = [[0.0, 0.0, 1.0, 1.0]]
        new_thought, thought_boxes = process_thought(thought)
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
            suffix = torch.tensor([ 3148,  1001, 29901,  1724,   338,   596,  2186,  1234, 29973,   319,
                     1799,  9047, 13566, 29901]).to(thought_ids.device)
            del(input_dict['conversation'])
            eoc_indices = [-1] + torch.where(thought_ids == eoc_id)[0].tolist() + [thought_ids.shape[0]-1]
            input_dict['input_ids'] = []
            for i in range(len(eoc_indices) - 1):
                input_dict['input_ids'].append(thought_ids[eoc_indices[i]+1:eoc_indices[i+1]+1])
                if i < len(eoc_indices) - 2:
                    input_dict['input_ids'].append(torch.tensor([img_id]).to(thought_ids.device))
            input_dict['input_ids'].append(suffix)
            input_dict['input_ids'] = torch.cat(input_dict['input_ids'])
        return input_dict

    def get_index(self, i):
        item = self.meta.iloc[i]
        return item['index'].item()

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = self.meta.iloc[i]
        question = item['question']
        image_path = byte2image(item['image'])
        options = get_options(item, all_options)
        hint = item['hint']
        if not is_none(hint):
            question = hint + '\n' + question

        for option_char, option in zip(all_options[:len(options)], options):
            question = question + '\n' + option_char + '. ' + option
        
        if self.require_cot:
            input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + prompt.format(question=question) + '\n' + COT_ACTIVATION
        else:
            input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question)
        
        if self.single_pred_prompt:
            input_sent = input_sent + '\n' + "Answer with the option's letter from the given choices directly."
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image_path], 'conversation': sources}

class MMBenchOptDataset(Dataset):
    def __init__(
        self,
        path: str,
        single_pred_prompt: bool=False,
        require_cot: bool=False,
        option_in_context: bool=False,
    ):
        self.path = path
        self.meta = pd.read_table(path)
        self.single_pred_prompt = single_pred_prompt
        self.require_cot = require_cot
        self.option_in_context = option_in_context
        self.naive_format = False
        self.txt_prompt = False
        self.use_image_path = False

    def __len__(self) -> int:
        return self.meta.shape[0]
    
    def getlabel(self, i):
        item = self.meta.iloc[i]
        return item['answer']

    def __getitem__(self, i: int) -> dict[str, Any]:
        item = self.meta.iloc[i]
        question = item['question']
        image_path = byte2image(item['image'])
        options = get_options(item, all_options)
        hint = item['hint']
        if not is_none(hint):
            question = hint + '\n' + question
        if self.option_in_context:
            question = question + ' Select from following options: ' + '; '.join([o.lower() for o in options]) + '.'
        if self.require_cot:
            if self.txt_prompt:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question) + ' ' + COT_ACTIVATION_TXT
            else:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + prompt.format(question=question) + ' ' + COT_ACTIVATION
        else:
            if not self.naive_format:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question)
            else:
                input_sent = prompt.format(question=question)
        
        if self.use_image_path:
            image_path = '/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/Evaluation/MMBench_DEV_images/image_{}.jpg'.format(i)
            image = image_path
        if self.naive_format:
            return {'input_images': [image_path], 'question': input_sent, 'options': options, 'item_id': 'eval_{}'.format(i)}
        
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image_path], 'conversation': sources, 'options': options}

class SEEDDataset(Dataset):
    def __init__(
        self,
        path: str,
        image_dir: str,
    ):
        self.meta = [item for item in json.load(open(path))['questions'] if item['data_type']=='image']
        self.choices = ['choice_a', 'choice_b', 'choice_c', 'choice_d']
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.meta) * len(self.choices)
    
    def getlabel(self, i):
        item_index = i // len(self.choices)
        choice_index = i - item_index * len(self.choices)
        item = self.meta[item_index]
        return item['answer']

    def get_index(self, i):
        item_index = i // len(self.choices)
        choice_index = i - item_index * len(self.choices)
        item = self.meta[item_index]
        return item['question_id'], ['A', 'B', 'C', 'D'][choice_index]

    def __getitem__(self, i: int) -> dict[str, Any]:
        item_index = i // len(self.choices)
        choice_index = i - item_index * len(self.choices)
        item = self.meta[item_index]
        choice_key = self.choices[choice_index]
        question = item['question']
        response_candidate = item[choice_key]
        image_path = os.path.join(self.image_dir, item['data_id'])
        image = Image.open(image_path).convert('RGB')
        
        input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question)
        
        sources = [{'from': 'human', 'value': input_sent}, {'from': 'gpt', 'value': response_candidate}]

        return {'input_images': [image], 'conversation': sources}

class SEEDOptionDataset(Dataset):
    def __init__(
        self,
        path: str,
        image_dir: str,
        require_cot: bool=False,
        option_in_context: bool=False,
    ):
        self.meta = [item for item in json.load(open(path))['questions'] if item['data_type']=='image']
        # self.meta = [item for i,item in enumerate(self.meta) if i in [267, 301, 334]]
        self.choices = ['choice_a', 'choice_b', 'choice_c', 'choice_d']
        self.image_dir = image_dir
        self.require_cot = require_cot
        self.option_in_context = option_in_context
        self.naive_format = False
        self.txt_prompt = False
        self.use_image_path = False

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        item = self.meta[i]
        return item['answer']

    def get_index(self, i):
        item_index = i // len(self.choices)
        choice_index = i - item_index * len(self.choices)
        item = self.meta[item_index]
        return item['question_id'], ['A', 'B', 'C', 'D'][choice_index]

    def __getitem__(self, i: int) -> dict[str, Any]:
        item_index = i
        item = self.meta[item_index]
        # choice_key = self.choices[choice_index]
        question = item['question']
        options = [item[k].lower() for k in self.choices]
        if self.option_in_context:
            question = question + ' Select from following options: ' + '; '.join(options) + '.'
        image_path = os.path.join(self.image_dir, item['data_id'])
        image = Image.open(image_path).convert('RGB')
        if self.require_cot:
            if self.txt_prompt:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question) + ' ' + COT_ACTIVATION_TXT
            else:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + prompt.format(question=question) + ' ' + COT_ACTIVATION
        else:
            if not self.naive_format:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question)
            else:
                input_sent = prompt.format(question=question)
        
        if self.use_image_path:
            image = image_path
        if self.naive_format:
            return {'input_images': [image], 'question': input_sent, 'options': options, 'item_id': 'eval_{}'.format(i)}

        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources, 'options': options}

class VStarDataset(Dataset):
    def __init__(
        self,
        path: str,
        image_dir: str,
        require_cot: bool=False,
        option_in_context: bool=False,
    ):
        self.meta = json.load(open(path))
        self.image_dir = image_dir
        self.require_cot = require_cot
        self.use_image_path = False
        self.naive_format = False
        self.option_in_context = option_in_context
        self.txt_prompt = False

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        return 0

    def __getitem__(self, i: int) -> dict[str, Any]:
        item_index = i
        item = self.meta[item_index]
        question = item['question']
        options = item["options"]
        image_path = os.path.join(self.image_dir, item['image'])
        image = Image.open(image_path).convert('RGB')

        if self.option_in_context:
            question = question + ' Select from following options: ' + '; '.join(options) + '.'
        
        if self.require_cot:
            if self.txt_prompt:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question) + ' ' + COT_ACTIVATION_TXT
            else:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + prompt.format(question=question) + ' ' + COT_ACTIVATION
        else:
            if not self.naive_format:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question)
            else:
                input_sent = prompt.format(question=question)
        
        if self.use_image_path:
            image = image_path
        if self.naive_format:
            return {'input_images': [image], 'question': input_sent, 'options': options, 'item_id': 'eval_{}'.format(i)}
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources, "options": options}


class WinoTextDataset(Dataset):
    def __init__(
        self,
        option_in_context: bool=False,
        require_cot: bool=False,
    ):
        self.meta = load_dataset('facebook/winoground', use_auth_token='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')['test']
        self.option_in_context = option_in_context
        self.require_cot = require_cot
        self.naive_format = False
        self.use_image_path = False
        self.txt_prompt = False

    def __len__(self) -> int:
        return len(self.meta) * 2
    
    def getlabel(self, i):
        return i % 2

    def __getitem__(self, i: int) -> dict[str, Any]:
        item_index = i // 2
        item = self.meta[item_index]
        # question = "Which event is happening in the image? \"{}\" or \"{}\"?".format(item['caption_0'], item['caption_1'])
        question = "Please describe the image."
        options = [item['caption_0'], item['caption_1']]
        if i % 2 == 0:
            k = 0
            image = item['image_0'].convert('RGB')
        else:
            k = 1
            image = item['image_1'].convert('RGB')
        if self.option_in_context:
            question = question + ' Select from following options: ' + '; '.join(options) + '.'
        
        if self.require_cot:
            if self.txt_prompt:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question) + ' Locate objects and in your response.'
            else:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + prompt.format(question=question) + ' Locate objects and provide bounding boxes in your response.'# ' ' + COT_ACTIVATION
        else:
            if not self.naive_format:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question)
            else:
                input_sent = prompt.format(question=question)
        
        if self.use_image_path:
            image_path = '/mnt/bn/yangmin-priv/luoruipu/data/multimodal-datasets/Evaluation/Wino/image_{}_{}.jpg'.format(item_index, k)
            image = image_path
        if self.naive_format:
            return {'input_images': [image], 'question': input_sent, 'options': options, 'item_id': 'eval_{}'.format(i)}
            # input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question)
        
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources, "options": options}

class EmbSpatialDataset(Dataset):
    def __init__(
        self,
        path: str,
        image_dir: str,
        option_in_context: bool=False,
        require_cot: bool=False,
    ):
        self.meta = json.load(open(path))
        self.meta = [item for item in self.meta if item['relation'] not in ['on top of', 'inside']]
        self.image_dir = image_dir
        self.require_cot = require_cot
        self.option_in_context = option_in_context
        self.naive_format = False
        self.use_image_path = False
        self.txt_prompt = False

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        return self.meta[i]['answer']

    def __getitem__(self, i: int) -> dict[str, Any]:
        item_index = i
        item = self.meta[item_index]
        question = item['question']
        options = item["answer_options"]
        image_path = os.path.join(self.image_dir, os.path.split(item['image'])[-1])
        image = Image.open(image_path).convert('RGB')
        if self.option_in_context:
            question = question + ' Select from following options: ' + '; '.join(options) + '.'
        
        if self.require_cot:
            if self.txt_prompt:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question) + ' ' + COT_ACTIVATION_TXT
            else:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + prompt.format(question=question) + ' ' + COT_ACTIVATION
        else:
            if not self.naive_format:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question)
            else:
                input_sent = prompt.format(question=question)
        
        if self.use_image_path:
            image = image_path
        if self.naive_format:
            return {'input_images': [image], 'question': input_sent, 'options': options, 'item_id': 'eval_{}'.format(i)}
        
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources, "options": options}

def reduce_answer(item):
    if item['answer'] in ['0','1','10','2','3','4','5','6','7','8','9']:
        q_type = 'count'
        space = [int(i) for i in ['0','1','10','2','3','4','5','6','7','8','9']]
        return [str(i) for i in list(sorted(space, key=lambda x: abs(x-int(item['answer']))))[:4]]
    elif item['answer'] in ['red', 'blue', 'brown','cyan', 'gray', 'green', 'purple', 'yellow']:
        q_type = 'color'
        space = [item['answer']] + random.sample([k for k in ['red', 'blue', 'brown','cyan', 'gray', 'green', 'purple', 'yellow'] if k!=item['answer']], 3)
        random.shuffle(space)
        return space
    elif item['answer'] in ['rubber', 'metal']:
        q_type = 'texture'
        return ['rubber', 'metal']
    elif item['answer'] in ['small', 'large']:
        q_type = 'size'
        return ['small', 'large']
    elif item['answer'] in ['yes', 'no']:
        q_type = 'binary'
        return ['yes', 'no']
    elif item['answer'] in ['cube', 'cylinder', 'sphere']:
        q_type = 'shape'
        return ['cube', 'cylinder', 'sphere']
    else:
        raise ValueError

def question_type(item):
    if item['answer'] in ['0','1','10','2','3','4','5','6','7','8','9']:
        q_type = 'count'
        space = [int(i) for i in ['0','1','10','2','3','4','5','6','7','8','9']]
    elif item['answer'] in ['red', 'blue', 'brown','cyan', 'gray', 'green', 'purple', 'yellow']:
        q_type = 'color'
        space = [item['answer']] + random.sample([k for k in ['red', 'blue', 'brown','cyan', 'gray', 'green', 'purple', 'yellow'] if k!=item['answer']], 3)
    elif item['answer'] in ['rubber', 'metal']:
        q_type = 'texture'
    elif item['answer'] in ['small', 'large']:
        q_type = 'size'
    elif item['answer'] in ['yes', 'no']:
        q_type = 'binary'
    elif item['answer'] in ['cube', 'cylinder', 'sphere']:
        q_type = 'shape'
    else:
        raise ValueError
    return q_type

class CLEVRDataset(Dataset):
    def __init__(
        self,
        path: str,
        image_dir: str,
        option_in_context: bool=False,
        require_cot: bool=False,
        naive_format: bool=False,
    ):
        self.meta = json.load(open(path))
        self.image_dir = image_dir
        self.require_cot = require_cot
        self.option_in_context = option_in_context
        self.naive_format = naive_format
        self.use_image_path = False
        self.txt_prompt = False

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        return self.meta[i]['answer_options'].index(self.meta[i]['answer'])

    def question_type(self, i):
        item = self.meta[i]
        if item['answer'] in ['0','1','10','2','3','4','5','6','7','8','9']:
            q_type = 'count'
            return q_type
        elif item['answer'] in ['red', 'blue', 'brown','cyan', 'gray', 'green', 'purple', 'yellow']:
            q_type = 'color'
        elif item['answer'] in ['rubber', 'metal']:
            q_type = 'texture'
        elif item['answer'] in ['small', 'large']:
            q_type = 'size'
        elif item['answer'] in ['yes', 'no']:
            q_type = 'binary'
        elif item['answer'] in ['cube', 'cylinder', 'sphere']:
            q_type = 'shape'
        else:
            raise ValueError
        return q_type

    def __getitem__(self, i: int) -> dict[str, Any]:
        item_index = i
        item = self.meta[item_index]
        question = item['question']
        options = item["answer_options"]
        image_path = os.path.join(self.image_dir, os.path.split(item['image_filename'])[-1])
        image = Image.open(image_path).convert('RGB')
        if self.option_in_context:
            question = question + ' Select from following options: ' + '; '.join(options) + '.'
        
        if self.require_cot:
            if self.txt_prompt:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question) + ' ' + COT_ACTIVATION_TXT
            else:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + prompt.format(question=question) + ' ' + COT_ACTIVATION
        else:
            if not self.naive_format:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question)
            else:
                input_sent = prompt.format(question=question)
        
        if self.use_image_path:
            image = image_path
        if self.naive_format:
            return {'input_images': [image], 'question': input_sent, 'options': options, 'item_id': 'eval_{}'.format(i)}
        
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources, "options": options}

class Visual7WDataset(Dataset):
    def __init__(
        self,
        path,
        base_path: str,
        require_cot: bool=False,
        expand2square: bool=True,
        phrase_format: str='text',
        phrase_prec: int=2,
        option_in_context: bool=False,
        object_format: str='representation',
    ):
        self.path = path
        if path.endswith('json'):
            self.meta = json.load(open(path))
        else:
            self.meta = [json.loads(line) for line in open(path)]
        self.base_path = base_path
        self.require_cot = require_cot
        self.naive_format = False
        self.expand2square = expand2square
        self.phrase_format = phrase_format
        self.phrase_prec = phrase_prec
        self.option_in_context = option_in_context
        self.object_format = object_format

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        item = self.meta[i]
        return item['candidates'].index(item['answer'])

    def proc_box(self, bbox, image):
        width, height = image.size
        new_box = [c/ width if i%2==0 else c/height for i,c in enumerate(bbox)]
        if self.expand2square:
            new_box = reshape_box(image, new_box)
        coor = box2str(new_box, self.phrase_format, self.phrase_prec)
        if self.naive_format:
            coor_format = '[' + coor + ']'
        elif self.object_format == 'text':
            coor_format = DEFAULT_BOC_TOKEN + coor + DEFAULT_EOC_TOKEN
        else:
            coor_format = DEFAULT_BOC_TOKEN + coor + DEFAULT_EOC_TOKEN + ALL_IMG_TOKENS_STR
        return new_box, coor_format
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        
        item = self.meta[i]
        question = item['question']
        image_path = os.path.join(self.base_path, item['file_path'])
        image = Image.open(image_path).convert("RGB")
        options_infos = [self.proc_box(b, image) for b in item['candidates']]
        if self.option_in_context:
            question = question + ' Select from the following options: ' + '; '.join([b[1] for b in options_infos]) + '.'
        options = [b[1].replace(ALL_IMG_TOKENS_STR, '') for b in options_infos]
        if self.object_format == 'representation' and not self.naive_format and self.option_in_context:
            box_info = [[0.0, 0.0, 1.0, 1.0]] + [b[0] for b in options_infos]
        else:
            box_info = [[0.0, 0.0, 1.0, 1.0]]
        if self.require_cot:
            input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + question + ' ' + COT_ACTIVATION
        else:
            # input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=(question + ' Please answer yes or no.'))
            if not self.naive_format:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + prompt.format(question=question)
            else:
                input_sent = prompt.format(question=question)
        # if self.describe:
        #     input_sent = ALL_IMG_TOKENS_STR + '\n' + question
            # input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + question + ' ' + 'Locate objects and provide bounding boxes in your response.'

        # image = Image.open(image_path).convert("RGB")

        if self.naive_format:
            return {'input_images': [image], 'question': input_sent, 'options': options, 'item_id': 'eval_{}'.format(i)}
        
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources, "options": options, 'box': box_info}