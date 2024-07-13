prompt = "{question}\nAnswer the question using a single word or phrase."

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
from ..utils.box_utils import *
from datasets import load_dataset

from constants import * # ALL_IMG_TOKENS_STR, DEFAULT_BOP_TOKEN, DEFAULT_EOP_TOKEN, DEFAULT_GRD_TOKEN


class MMEDataset(Dataset):
    def __init__(
        self,
        path: str,
        base_path: str,
        require_cot: bool=False
    ):
        self.path = path
        self.meta = json.load(open(path))
        self.base_path = base_path
        self.require_cot = require_cot

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        return self.meta[i]['answer']

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
                suffix = torch.tensor([  733, 16289, 28793,  1824,   349,   574,  1480,  4372, 28804,   733,
                                        28748, 16289, 28793]).to(thought_ids.device)
            else:
                suffix = torch.tensor([3148, 1001, 29901, 1724, 338, 596, 2186, 1234, 29973, 13, 3529, 1234, 4874, 470, 694, 29889, 319, 1799, 9047, 13566, 29901]).to(thought_ids.device)
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
        question = item['question']
        image_path = os.path.join(self.base_path, item['image'])
        if self.require_cot:
            input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + question.replace(' Please answer yes or no.', '') + ' ' + COT_ACTIVATION
        else:
            input_sent = ALL_IMG_TOKENS_STR + ' ' + prompt.format(question=question)

        image = Image.open(image_path).convert("RGB")
        
        # image_1 = Image.fromarray(image_1)
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources}

class VQAv2Dataset(Dataset):
    def __init__(
        self,
        path: str,
        base_path: str,
        require_cot: bool=False,
    ):
        self.path = path
        self.meta = [json.loads(line.strip()) for line in open(path)]
        self.base_path = base_path
        self.require_cot = require_cot

    def __len__(self) -> int:
        return len(self.meta)

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
                suffix = torch.tensor([  733, 16289, 28793,  1824,   349,   574,  1480,  4372, 28804,   733,
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
        image_fn = item['image']
        split = image_fn.split('_')[1]
        question = item['text']
        image_path = os.path.join(self.base_path, split, item['image'])
        if self.require_cot:
            input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + ' ' + question.replace('\nAnswer the question using a single word or phrase.', '') + ' ' + COT_ACTIVATION
        else:
            input_sent = ALL_IMG_TOKENS_STR +' '+ question

        image = Image.open(image_path).convert("RGB")
        
        # image_1 = Image.fromarray(image_1)
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources}

class VizWizDataset(Dataset):
    def __init__(
        self,
        path: str,
        base_path: str,
    ):
        self.path = path
        self.meta = json.load(open(path))
        self.base_path = base_path
        self.prompt = "{}\nWhen the provided information is insufficient, respond with 'Unanswerable'.\nAnswer the question using a single word or phrase."

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, i: int) -> dict[str, Any]:
        
        item = self.meta[i]
        question = item['question']
        image_path = os.path.join(self.base_path, item['image'])
        input_sent = ALL_IMG_TOKENS_STR +' '+ self.prompt.format(question)

        image = Image.open(image_path).convert("RGB")
        
        # image_1 = Image.fromarray(image_1)
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources}

class GQADataset(Dataset):
    def __init__(
        self,
        path: str,
        base_path: str,
        require_cot: bool=False,
        short_prompt: bool=False,
    ):
        self.path = path
        self.meta = json.load(open(path))
        self.keys = list(self.meta.keys())
        self.base_path = base_path
        self.require_cot = require_cot
        self.short_prompt = short_prompt
        self.prompt = "{}\nAnswer the question using a single word or phrase."

    def __len__(self) -> int:
        return len(self.keys)

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
                if self.short_prompt:
                    suffix = torch.tensor([733, 16289, 28793, 1824, 349, 574, 1480, 4372, 28804, 13, 2820, 16981, 272, 2996, 1413, 264, 2692, 1707, 
                                            442, 14804, 28723, 733, 28748, 16289, 28793]).to(thought_ids.device)
                else:
                    suffix = torch.tensor([  733, 16289, 28793,  1824,   349,   574,  1480,  4372, 28804,   733,
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
                    # check if image already appears
                    if thought_ids[eoc_indices[i+1]+1].item() != img_id:
                        input_dict['input_ids'].append(torch.tensor([img_id]).to(thought_ids.device))
            input_dict['input_ids'].append(suffix)
            input_dict['input_ids'] = torch.cat(input_dict['input_ids'])
        return input_dict

    def __getitem__(self, i: int) -> dict[str, Any]:
        
        key = self.keys[i]
        item = self.meta[key]
        question = item['question']
        image_path = os.path.join(self.base_path, '{}.jpg'.format(item['imageId']))
        if self.require_cot:
            input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + question + ' ' + COT_ACTIVATION
        else:
            input_sent = ALL_IMG_TOKENS_STR +' '+ self.prompt.format(question)

        image = Image.open(image_path).convert("RGB")
        
        # image_1 = Image.fromarray(image_1)
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources}

class TextVQADataset(Dataset):
    def __init__(
        self,
        path: str,
        base_path: str,
        require_cot: bool=False,
    ):
        self.path = path
        self.meta = [json.loads(line.strip()) for line in open(path)]
        self.base_path = base_path
        self.require_cot = require_cot

    def __len__(self) -> int:
        return len(self.meta)

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
        image_fn = item['image']
        question = item['text']
        image_path = os.path.join(self.base_path, item['image'])
        if self.require_cot:
            input_sent = DEFAULT_GRD_TOKEN + '\n' + question.replace('Answer the question using a single word or phrase.', COT_ACTIVATION)
        else:
            input_sent = ALL_IMG_TOKENS_STR +'\n'+ question

        image = Image.open(image_path).convert("RGB")
        
        # image_1 = Image.fromarray(image_1)
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources}


class VSRDataset(Dataset):
    def __init__(
        self,
        path: str,
        base_path: str,
        require_cot: bool=False,
        options: bool=False,
    ):
        self.path = path
        self.meta = [json.loads(line) for line in open(path)]
        self.base_path = base_path
        self.require_cot = require_cot
        self.prompt = "Is there an event \"{}\" taking place in the image?"
        # self.prompt = "Please verify whether \"{}\"."
        self.options = options
        self.naive_format = False
        self.use_image_path = False
        self.txt_prompt = False

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        return self.meta[i]['label']

    def cot_turn(self, i, thought=None, thought_ids=None, eoc_id=None, img_id=None, mistral=False, sub_image_bind=False):
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
        if sub_image_bind:
            cache_raw_image = expand2square_fn(input_dict['input_images'][0], (0,0,0))
            w, h = cache_raw_image.size
            new_sub_images = []
            for b in thought_boxes:
                x_min, y_min, x_max, y_max = b
                x_min = x_min*w
                y_min = y_min*h
                x_max = x_max*w
                y_max = y_max*h
                sub_image = cache_raw_image.crop((x_min, y_min, x_max, y_max))
                new_sub_images.append(resize_image_to_square(sub_image))
            num_images = len(new_sub_images) + 1
            input_dict['input_images'] = input_dict['input_images'] + new_sub_images
            all_box = [[0.0, 0.0, 1.0, 1.0]]*num_images
        input_dict['box'] = all_box
        if thought_ids is not None:
            thought_ids = thought_ids.squeeze()
            if mistral:
                if self.txt_prompt:
                    suffix = torch.tensor([   733, 16289, 28793,  1824,   349,   574,  1480,  4372, 28804, 5919, 4372, 5081, 442, 708, 28723,  733,
                                            28748, 16289, 28793]).to(thought_ids.device)
                else:
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
        question = item['caption']
        split = item['image_link'].split('/')[-2]
        image_path = os.path.join(self.base_path, split, item['image'])
        if self.require_cot:
            if self.txt_prompt:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + self.prompt.format(question.lower()) + ' ' + COT_ACTIVATION_TXT
            else:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + self.prompt.format(question.lower()) + ' ' + COT_ACTIVATION
        else:
            if not self.naive_format:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=(self.prompt.format(question.lower()) + ' Please answer yes or no.'))
            else:
                input_sent = prompt.format(question=(self.prompt.format(question.lower()) + ' Please answer yes or no.'))

        image = Image.open(image_path).convert("RGB")
        if self.use_image_path:
            image = image_path

        if self.naive_format:
            return {'input_images': [image], 'question': input_sent, 'options': ['no', 'yes'], 'item_id': 'eval_{}'.format(i)}
        
        # image_1 = Image.fromarray(image_1)
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources, 'options': ['no', 'yes']}

class WhoopsDataset(Dataset):
    def __init__(
        self,
        require_cot: bool=False
    ):
        self.meta = load_dataset('nlphuji/whoops', use_auth_token='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')['test']
        self.require_cot = require_cot
        self.index2base = {}
        index = 0
        for i in range(len(self.meta)):
            item = self.meta[i]
            for j,qa in enumerate(item['question_answering_pairs']):
                self.index2base[index] = (i, j)
                index += 1
        self.total_num = index

    def __len__(self) -> int:
        return self.total_num
    
    def getlabel(self, i):
        i, j = self.index2base[i]
        return self.meta[i]['question_answering_pairs'][j][1]

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
        
        index_base, index_ques = self.index2base[i]
        item = self.meta[index_base]
        question = item['question_answering_pairs'][index_ques][0]
        image = item['image']
        if self.require_cot:
            input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + question + ' ' + COT_ACTIVATION
        else:
            input_sent = ALL_IMG_TOKENS_STR + '\n' + question

        # image = Image.open(image_path).convert("RGB")
        
        # image_1 = Image.fromarray(image_1)
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources}

class POPEDataset(Dataset):
    def __init__(
        self,
        path,
        base_path: str,
        list_path: bool=False,
        require_cot: bool=False
    ):
        self.path = path
        if list_path:
            self.meta = []
            for p in path:
                self.meta.extend([json.loads(line) for line in open(p)])
        else:
            self.meta = [json.loads(line) for line in open(path)]
        self.base_path = base_path
        self.require_cot = require_cot
        self.naive_format = False
        self.use_image_path = False
        self.txt_prompt = False

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        return self.meta[i]['label']

    def cot_turn(self, i, thought=None, thought_ids=None, eoc_id=None, img_id=None, mistral=False, sub_image_bind=False):
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
        if sub_image_bind:
            cache_raw_image = expand2square_fn(input_dict['input_images'][0], (0,0,0))
            w, h = cache_raw_image.size
            new_sub_images = []
            for b in thought_boxes:
                x_min, y_min, x_max, y_max = b
                x_min = x_min*w
                y_min = y_min*h
                x_max = x_max*w
                y_max = y_max*h
                sub_image = cache_raw_image.crop((x_min, y_min, x_max, y_max))
                new_sub_images.append(resize_image_to_square(sub_image))
            num_images = len(new_sub_images) + 1
            input_dict['input_images'] = input_dict['input_images'] + new_sub_images
            all_box = [[0.0, 0.0, 1.0, 1.0]]*num_images
        input_dict['box'] = all_box
        if thought_ids is not None:
            thought_ids = thought_ids.squeeze()
            if mistral:
                if self.txt_prompt:
                    suffix = torch.tensor([   733, 16289, 28793,  1824,   349,   574,  1480,  4372, 28804, 5919, 4372, 5081, 442, 708, 28723,  733,
                                            28748, 16289, 28793]).to(thought_ids.device)
                else:
                    suffix = torch.tensor([   733, 16289, 28793,  1824,   349,   574,  1480,  4372, 28804,   733,
                                            28748, 16289, 28793]).to(thought_ids.device)
                # suffix = torch.tensor([   733, 16289, 28793,  1824,   349,   574,  1480,  4372, 28804,   733,
                #                         28748, 16289, 28793]).to(thought_ids.device)
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
        question = item['text']
        split = item['image'].split('_')[1]
        image_path = os.path.join(self.base_path, split, item['image'])
        if self.require_cot:
            if self.txt_prompt:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + question + ' ' + COT_ACTIVATION_TXT
            else:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + question + ' ' + COT_ACTIVATION
        else:
            # input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=(question + ' Please answer yes or no.'))
            if not self.naive_format:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question + ' Please answer yes or no.')
            else:
                input_sent = prompt.format(question=(question + ' Please answer yes or no.'))

        image = Image.open(image_path).convert("RGB")
        if self.use_image_path:
            image = image_path

        if self.naive_format:
            return {'input_images': [image], 'question': input_sent, 'item_id': 'eval_{}'.format(i)}
        
        # image_1 = Image.fromarray(image_1)
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources}

class HallusionBenchDataset(Dataset):
    def __init__(
        self,
        path: str,
        base_path: str,
        require_cot: bool=False
    ):
        self.path = path
        self.meta = [item for item in json.load(open(path)) if item['visual_input']=='1']
        self.base_path = base_path
        self.require_cot = require_cot

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        return 'yes' if self.meta[i]['gt_answer'] == '1' else 'no'

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
                suffix = torch.tensor([3148, 1001, 29901, 1724, 338, 596, 2186, 1234, 29973, 13, 3529, 1234, 4874, 470, 694, 29889, 319, 1799, 9047, 13566, 29901]).to(thought_ids.device)
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
        question = item['question']
        img_fn = item['filename'].lstrip('./')
        image_path = os.path.join(self.base_path, img_fn)
        if self.require_cot:
            input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + question + ' ' + COT_ACTIVATION
        else:
            if not self.naive_format:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question + ' Please answer yes or no.')
            else:
                input_sent = prompt.format(question=(question + ' Please answer yes or no.'))

        image = Image.open(image_path).convert("RGB")

        if self.naive_format:
            return {'input_images': [image], 'question': input_sent, 'item_id': 'eval_{}'.format(i)}
        
        # image_1 = Image.fromarray(image_1)
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources, 'options': ['yes', 'no']}


class AMBERDataset(Dataset):
    def __init__(
        self,
        path,
        base_path: str,
        list_path: bool=False,
        require_cot: bool=False,
        describe: bool=False,
    ):
        self.path = path
        if list_path:
            self.meta = []
            for p in path:
                self.meta.extend(json.load(open(p)))
        else:
            self.meta = json.load(open(path))
        self.base_path = base_path
        self.require_cot = require_cot
        self.naive_format = False
        self.describe = describe
        self.txt_prompt = False

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        return self.meta[i]['id']

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
        question = item['query']
        image_path = os.path.join(self.base_path, item['image'])
        if self.require_cot:
            if self.describe:
                if self.txt_prompt:
                    input_sent = ALL_IMG_TOKENS_STR + '\n' + question + ' ' + COT_ACTIVATION_TXT
                else:
                    input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + question + ' ' + COT_ACTIVATION # 'Locate objects and provide bounding boxes in your response.'
            else:
                input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + question + ' ' + COT_ACTIVATION
        else:
            # input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=(question + ' Please answer yes or no.'))
            if not self.naive_format:
                input_sent = ALL_IMG_TOKENS_STR + '\n' + prompt.format(question=question)
            else:
                input_sent = prompt.format(question=question)
        # if self.describe:
        #     input_sent = ALL_IMG_TOKENS_STR + '\n' + question
            # input_sent = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + question + ' ' + 'Locate objects and provide bounding boxes in your response.'

        image = Image.open(image_path).convert("RGB")

        if self.naive_format:
            return {'input_images': [image], 'question': input_sent, 'item_id': 'eval_{}'.format(i)}
        
        # image_1 = Image.fromarray(image_1)
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources}

class LookTwiceDataset(Dataset):
    def __init__(
        self,
        path,
        base_path: str,
        list_path: bool=False,
        require_cot: bool=False,
        expand2square: bool=True,
        phrase_format: str='text',
        phrase_prec: int=2,
    ):
        self.path = path
        self.meta = json.load(open(path))
        self.base_path = base_path
        self.require_cot = require_cot
        self.naive_format = False
        self.expand2square = expand2square
        self.phrase_format = phrase_format
        self.phrase_prec = phrase_prec

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        return self.meta[i]['answer']

    def cot_turn(self, i, thought=None, thought_ids=None, eoc_id=None, img_id=None, mistral=False):
        input_dict = self[i]
        all_box = input_dict['box']
        new_thought, thought_boxes = process_thought(thought, mistral=mistral)
        all_box.extend(thought_boxes)
        # print(all_box)
        input_dict['conversation'].append(
            {'from': 'gpt', 'value': new_thought}
        )
        input_dict['conversation'].append(
            {'from': 'human', 'value': 'What is your final answer?'}
        )
        input_dict['box'] = torch.tensor(all_box, dtype=torch.float32)
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

    def proc_question(self, question, bbox, image):
        txt_splits = question.split('these')
        assert len(txt_splits) == 2
        width, height = image.size
        new_box = [c/ width if i%2==0 else c/height for i,c in enumerate(bbox)]
        if self.expand2square:
            new_box = reshape_box(image, new_box)
        coor = box2str(new_box, self.phrase_format, self.phrase_prec)
        coor_format = DEFAULT_BOC_TOKEN + coor + DEFAULT_EOC_TOKEN + ALL_IMG_TOKENS_STR
        return txt_splits[0] + 'these' + coor_format + txt_splits[1], new_box
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        
        item = self.meta[i]
        question = item['question']
        image_path = os.path.join(self.base_path, item['image'])
        image = Image.open(image_path).convert("RGB")
        question, box = self.proc_question(question, item['bbox'], image)
        box_info = [[0.0, 0.0, 1.0, 1.0], box]
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
            return {'input_images': [image], 'question': input_sent, 'item_id': 'eval_{}'.format(i)}
        
        # image_1 = Image.fromarray(image_1)
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources, 'box':box_info}


class VGLocalDataset(Dataset):
    def __init__(
        self,
        path,
        base_path: str,
        list_path: bool=False,
        require_cot: bool=False,
        expand2square: bool=True,
        phrase_format: str='text',
        phrase_prec: int=2,
        object_format: str='representation'
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
        self.object_format = object_format

    def __len__(self) -> int:
        return len(self.meta)
    
    def getlabel(self, i):
        return self.meta[i]['answer']

    def cot_turn(self, i, thought=None, thought_ids=None, eoc_id=None, img_id=None, mistral=False):
        input_dict = self[i]
        all_box = input_dict['box']
        new_thought, thought_boxes = process_thought(thought, mistral=mistral)
        all_box.extend(thought_boxes)
        # print(all_box)
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

    def proc_question(self, question, bbox, image):
        txt_splits = question[:-1]
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
        return txt_splits + coor_format + '?', new_box
    
    def __getitem__(self, i: int) -> dict[str, Any]:
        
        item = self.meta[i]
        question = item['question']
        image_path = os.path.join(self.base_path, item['file_path'].replace('images2/', '').replace('images/', ''))
        image = Image.open(image_path).convert("RGB")
        question, box = self.proc_question(question, item['bbox'], image)
        if self.object_format == 'representation' and not self.naive_format:
            box_info = [[0.0, 0.0, 1.0, 1.0], box]
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
            return {'input_images': [image], 'question': input_sent, 'item_id': 'eval_{}'.format(i)}
        
        # image_1 = Image.fromarray(image_1)
        sources = [{'from': 'human', 'value': input_sent}]

        return {'input_images': [image], 'conversation': sources, 'box':box_info}