from torch.utils.data import Dataset
import json
import os
import random
import copy
from PIL import Image
from constants import *
from ..utils.box_utils import box2str, reshape_box, resize_image_to_square
import math

class CoTQADataset(Dataset):
    def __init__(
        self,
        path: str,
        image_path: str = None,
        avoid_image_gen: bool = False,
        phrase_format: str='special_tokens',
        phrase_prec: int=2,
        phrase_space: bool=False,
        ignore_object: bool=False,
        expand2square: bool=False,
        object_format: str='image',
        further_instruct: bool=False,
    ):
        self.path = path
        self.avoid_image_gen = avoid_image_gen
        self.phrase_format = phrase_format
        self.phrase_prec = phrase_prec
        self.ignore_object = ignore_object
        self.phrase_space = phrase_space
        self.expand2square = expand2square
        self.object_format = object_format
        self.image_path = image_path
        self.further_instruct = further_instruct
        # loading the data
        self.meta = [json.loads(line) for line in open(self.path)]
        print('Shikra-COT has {} samples'.format(len(self)))

    def proc_box(self, box, image):
        w, h = image.size
        new_box = [c / w if (i%2==0) else c / h for i,c in enumerate(box)]
        x_min, y_min, x_max, y_max = new_box
        x_min = x_min*w
        y_min = y_min*h
        x_max = x_max*w
        y_max = y_max*h
        sub_image = image.crop((x_min, y_min, x_max, y_max))
        sub_image = resize_image_to_square(sub_image)
        if self.expand2square:
            new_box = reshape_box(image, new_box)
        return new_box, sub_image

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
        item = self.meta[index]
        image = item['img_path']
        assert self.object_format in ['representation', 'coordinate', 'text', 'image']
        image = Image.open(os.path.join(self.image_path, image)).convert('RGB')
        conversation = []

        # all box information
        box_infos = []
        cand_sub_images = []
        for b in item['boxes']:
            sub_info = self.proc_box(b, image)
            box_infos.append(sub_info[0])
            cand_sub_images.append(sub_info[1])
        # box_infos = [self.proc_box(b, image) for b in item['boxes']]
        all_box_input = [[0.0, 0.0, 1.0, 1.0]]
        image_label_masks = [0]
        all_sub_images = []

        # process the input and output
        input_query = item['question'].replace('<ph_st>', '').split('<ph_ed>')
        if self.object_format == 'text':
            refined_query = ALL_IMG_TOKENS_STR + '\n'
        else:
            refined_query = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n'
        for box_ind, current_box in enumerate(item['question_boxes_seq']):
            all_sub_images.extend([cand_sub_images[c] for c in current_box])
            current_box = [box_infos[c] for c in current_box]
            all_box_input.extend(current_box)
            image_label_masks.extend([0]*len(current_box))
            box_in_str = DEFAULT_SEP_TOKEN.join([DEFAULT_BOC_TOKEN + box2str(c, mode=self.phrase_format, prec=self.phrase_prec, space=self.phrase_space) + DEFAULT_EOC_TOKEN + ALL_IMG_TOKENS_STR for c in current_box])
            if self.object_format == 'text':
                box_in_str = ''
            elif self.object_format == 'coordinate':
                box_in_str = box_in_str.replace(ALL_IMG_TOKENS_STR, '')
            refined_query = refined_query + input_query[box_ind] + box_in_str
        
        if self.object_format == 'text':
            refined_query = refined_query + input_query[-1] + ' ' + COT_ACTIVATION_TXT
            refined_query = refined_query.replace('  ', '')
        else:
            refined_query = refined_query + input_query[-1] + ' ' + COT_ACTIVATION


        response = item['cot_with_ans'].replace('<ph_st>', '').split('<ph_ed>')
        refined_response = ''
        for box_ind, current_box in enumerate(item['answer_boxes_seq']):
            all_sub_images.extend([cand_sub_images[c] for c in current_box])
            current_box = [box_infos[c] for c in current_box]
            all_box_input.extend(current_box)
            image_label_masks.extend([0]*len(current_box))
            box_in_str = DEFAULT_SEP_TOKEN.join([DEFAULT_BOC_TOKEN + box2str(c, mode=self.phrase_format, prec=self.phrase_prec, space=self.phrase_space) + DEFAULT_EOC_TOKEN + ALL_IMG_TOKENS_STR for c in current_box])
            if self.object_format == 'text':
                box_in_str = ''
            elif self.object_format == 'coordinate':
                box_in_str = box_in_str.replace(ALL_IMG_TOKENS_STR, '')
            refined_response = refined_response + response[box_ind] + box_in_str
        refined_response = refined_response + response[-1]
        if self.object_format == 'text':
            refined_response = refined_response.replace('  ', ' ')

        conversation = [
            {'from': 'human', 'value': refined_query},
            {'from': 'gpt', 'value': refined_response}
        ]

        # if need further include short answer
        # only for GQA-COT and GPT4GEN_BoxCoT
        if self.further_instruct:
            conversation.extend([
                {'from': 'human', 'value': 'What is your final answer?'},
                {'from': 'gpt', 'value': item['answer']}]
            )
        if self.object_format == 'image':
            return {'input_images': [image] + all_sub_images if not self.ignore_object else [image], 'conversation': conversation, 'image_label_masks': image_label_masks}
        elif self.object_format == 'representation':
            return {'input_images': [image] , 'conversation': conversation, 'image_label_masks': image_label_masks, 'box': all_box_input}
        elif self.object_format in ['coordinate','text']:
            return {'input_images': [image] , 'conversation': conversation, 'image_label_masks': image_label_masks, 'box': all_box_input[:1]}
        else:
            raise ValueError

class GQACoTDataset(Dataset):
    def __init__(
        self,
        path: str,
        image_path: str = None,
        avoid_image_gen: bool = False,
        phrase_format: str='special_tokens',
        phrase_prec: int=2,
        phrase_space: bool=False,
        ignore_object: bool=False,
        expand2square: bool=False,
        object_format: str='image',
        further_instruct: bool=False,
        sample_weight: float=1.0,
        no_loading_image: bool=False,
    ):
        self.path = path
        self.avoid_image_gen = avoid_image_gen
        self.phrase_format = phrase_format
        self.phrase_prec = phrase_prec
        self.ignore_object = ignore_object
        self.phrase_space = phrase_space
        self.expand2square = expand2square
        self.object_format = object_format
        self.image_path = image_path
        self.no_loading_image = no_loading_image
        self.further_instruct = further_instruct
        self.sample_weight = sample_weight
        # loading the data
        self.meta = [json.loads(line) for line in open(self.path)]
        print('GQA-COT has {} samples'.format(len(self)))

    def proc_box(self, box, image):
        w, h = image.size
        new_box = [c / w if (i%2==0) else c / h for i,c in enumerate(box)]
        x_min, y_min, x_max, y_max = new_box
        x_min = x_min*w
        y_min = y_min*h
        x_max = x_max*w
        y_max = y_max*h
        try:
            sub_image = image.crop((x_min, y_min, x_max, y_max))
        except:
            sub_image = Image.new(image.mode, (32,32), (0,0,0))
        sub_image = resize_image_to_square(sub_image)
        if self.expand2square:
            new_box = reshape_box(image, new_box)
        return new_box, sub_image

    def __len__(self):
        return int(len(self.meta)*self.sample_weight)
    
    def __getitem__(self, i):
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)
        item = self.meta[i]
        image = item['imageId']
        assert self.object_format in ['representation', 'coordinate', 'text', 'image']
        image = Image.open(os.path.join(self.image_path, '{}.jpg'.format(image))).convert('RGB')
        conversation = []

        # all box information
        box_infos = []
        cand_sub_images = []
        for b in item['cot']['boxes']:
            sub_info = self.proc_box(b, image)
            box_infos.append(sub_info[0])
            cand_sub_images.append(sub_info[1])
        # box_infos = [self.proc_box(b, image) for b in item['cot']['boxes']]
        all_box_input = [[0.0, 0.0, 1.0, 1.0]]
        image_label_masks = [0]
        all_sub_images = []

        # process the input and output
        if self.object_format == 'text':
            refined_query = ALL_IMG_TOKENS_STR + '\n' + item['question'] + ' ' + COT_ACTIVATION_TXT
            refined_query = refined_query.replace('  ', ' ')
        else:
            refined_query = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + item['question'] + ' ' + COT_ACTIVATION

        response = item['cot']['value'].replace('<ph_st>', '').split('<ph_ed>')
        refined_response = ''
        for box_ind, current_box in enumerate(item['cot']['seq']):
            all_sub_images.extend([cand_sub_images[c] for c in current_box])
            current_box = [box_infos[c] for c in current_box]
            all_box_input.extend(current_box)
            image_label_masks.extend([0]*len(current_box))
            box_in_str = DEFAULT_SEP_TOKEN.join([DEFAULT_BOC_TOKEN + box2str(c, mode=self.phrase_format, prec=self.phrase_prec, space=self.phrase_space) + DEFAULT_EOC_TOKEN + ALL_IMG_TOKENS_STR for c in current_box])
            if self.object_format == 'text':
                box_in_str = ''
            elif self.object_format == 'coordinate':
                box_in_str = box_in_str.replace(ALL_IMG_TOKENS_STR, '')
            refined_response = refined_response + response[box_ind] + box_in_str
        refined_response = refined_response + response[-1]
        if self.object_format == 'text':
            refined_response = refined_response.replace('  ', ' ')

        conversation = [
            {'from': 'human', 'value': refined_query},
            {'from': 'gpt', 'value': refined_response}
        ]

        # if need further include short answer
        # only for GQA-COT and GPT4GEN_BoxCoT
        if self.further_instruct:
            conversation.extend([
                {'from': 'human', 'value': 'What is your final answer?'},
                {'from': 'gpt', 'value': item['answer']}]
            )
        if self.object_format == 'image':
            return {'input_images': [image] + all_sub_images if not self.ignore_object else [image], 'conversation': conversation, 'image_label_masks': image_label_masks}
        elif self.object_format == 'representation':
            return {'input_images': [image] , 'conversation': conversation, 'image_label_masks': image_label_masks, 'box': all_box_input}
        elif self.object_format in ['text', 'coordinate']:
            return {'input_images': [image] , 'conversation': conversation, 'image_label_masks': image_label_masks, 'box': all_box_input[:1]}
        else:
            raise ValueError
