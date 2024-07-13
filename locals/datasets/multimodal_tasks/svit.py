from torch.utils.data import Dataset
import json as js
import math
import random
import os
from PIL import Image
from constants import *
import re, copy
from ..utils.box_utils import box2str

class SVITDataset(Dataset):

    def __init__(self, 
                 path: str,
                 image_folder: str,
                 instruct: bool = False,
                 min_resize_res: int = 256,
                 max_resize_res: int = 256,
                 crop_res: int = 256,
                 flip_prob: float = 0.5,
                 sample_weight: float = 1.0,
                 check: bool = False,
                 output_mode: str = 'text',
                 shuffle: bool = False,
                 raw_image: bool = False,
                 inference: bool = False,
                 min_size: int = 50, 
                 avoid_image_gen: bool = False,
                 phrase_format: str='special_tokens',
                 phrase_prec: int=2,
                 no_process: int=False,
                 expand2square: bool=False,
                 object_format: str='image',
                 **kwargs):
        # load from json ../datasets/gqa-inpaint/meta_info.json
        self.path = path
        self.instruct = instruct
        self.inference = inference
        self.meta = js.load(open(path))
        self.image_folder = image_folder
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.no_process = no_process
        self.check = check
        self.raw_image = raw_image
        self.output_mode = output_mode
        self.shuffle = shuffle
        self.min_size = min_size
        self.avoid_image_gen = avoid_image_gen
        self.phrase_format = phrase_format
        self.phrase_prec = phrase_prec
        self.expand2square = expand2square
        self.object_format = object_format

        self.flip_prob = flip_prob
        self.sample_weight = sample_weight
        print(f"SVIT has {len(self)} samples!!")

    def __len__(self):
        return int(len(self.meta) * self.sample_weight)

    def naive_boxes(self, value, image):
        pattern = '\[[0-1]\.[0-9]+, [0-1]\.[0-9]+, [0-1]\.[0-9]+, [0-1]\.[0-9]+\]'
        all_phrases = re.findall(pattern, value)
        new_value = copy.deepcopy(value)
        return [eval(phrase) for phrase in all_phrases]
    
    def detect_boxes(self, value, image):
        pattern = '\[[0-1]\.[0-9]+, [0-1]\.[0-9]+, [0-1]\.[0-9]+, [0-1]\.[0-9]+\]'
        all_phrases = re.findall(pattern, value)
        new_value = copy.deepcopy(value)
        width, height = image.size
        all_sub_images = []
        all_box_info = []
        for phrase in all_phrases:
            box = eval(phrase)
            if self.expand2square:
                pass
            else:
                # need reshape
                raise NotImplementedError
            x_min, y_min, x_max, y_max = box
            all_box_info.append(box)
            x_min = x_min*width
            y_min = y_min*height
            x_max = x_max*width
            y_max = y_max*height
            sub_image = image.crop((x_min, y_min, x_max, y_max))
            all_sub_images.append(sub_image)
            # coor = [round(c*(LOC_TOKEN_NUM-1)) for c in box]
            # coor = ''.join([ALL_LOC_TOKENS[i] for i in coor])
            coor = box2str(box, self.phrase_format, self.phrase_prec)
            coor_format = DEFAULT_BOC_TOKEN + coor + DEFAULT_EOC_TOKEN + ALL_IMG_TOKENS_STR
            new_value = new_value.replace(phrase, coor_format)
        return new_value, all_sub_images, all_box_info

    def process_vg(self, conversation, image):
        num_rounds = len(conversation) // 2
        all_sub_images = []
        all_sub_image_masks = []
        new_conversation = copy.deepcopy(conversation)
        valid_conversation = []
        valid_index = []
        all_box_infos = []
        for i in range(num_rounds):
            from_human = conversation[2*i]['value']
            from_gpt = conversation[2*i + 1]['value']
            new_value, sub_images, box_info = self.detect_boxes(from_human, image)
            if len(sub_images) > 0:
                # if from bounding boxes to phrases
                new_conversation[2*i]['value'] = new_value
                all_sub_image_masks.extend([0]*len(sub_images))
                if from_gpt.endswith('.'):
                    new_from_gpt = DEFAULT_BOP_TOKEN + from_gpt[:-1] + DEFAULT_EOP_TOKEN + '.'
                else:
                    new_from_gpt = DEFAULT_BOP_TOKEN + from_gpt + DEFAULT_EOP_TOKEN
                new_conversation[2*i + 1]['value'] = new_from_gpt
            else:
                # from phrases to bounding boxes
                tmp_phrase = from_human.split(':')[-1]
                if tmp_phrase.endswith('.'):
                    tmp_phrase = tmp_phrase[:-1]
                new_conversation[2*i]['value'] = from_human.replace(tmp_phrase, DEFAULT_BOP_TOKEN + tmp_phrase + DEFAULT_EOP_TOKEN)
                new_value, sub_images, box_info = self.detect_boxes(from_gpt, image)
                new_conversation[2*i+1]['value'] = new_value
                if self.avoid_image_gen:
                    all_sub_image_masks.extend([0]*len(sub_images))
                else:
                    all_sub_image_masks.extend([1]*len(sub_images))
                
            w,h = sub_images[0].size
            if w >= self.min_size and h >= self.min_size:
                valid_index.append(i)
            all_sub_images.extend(sub_images)
            all_box_infos.extend(box_info)
        # print(valid_index)
        # print(all_sub_images)
        for i,ind in enumerate(valid_index):
            if i == 0:
                if ind != 0:
                    tmp = new_conversation[2*ind]['value']
                    new_conversation[2*ind]['value'] = '<image>\n'+tmp
            valid_conversation.append(new_conversation[2*ind])
            valid_conversation.append(new_conversation[2*ind + 1])
        return valid_conversation, [all_sub_images[i] for i in valid_index], [all_sub_image_masks[i] for i in valid_index], [[0.0, 0.0, 1.0, 1.0]]+[all_box_infos[i] for i in valid_index]
    
    def get_image_fn(self, i):
        item = self.meta[i]
        if 'image' in item:
            image_fn = item['image']
            dirname = image_fn.split('/')[0]
            image_fn = '/'.join(image_fn.split('/')[1:])
            all_sub_images = []
            all_sub_image_masks = []
            if dirname == 'vg':
                dirname = 'VG'
            elif dirname == 'coco':
                dirname = 'COCO2017'
            elif dirname == 'textvqa':
                dirname = 'TextVQA'
            elif dirname == 'gqa':
                dirname = 'GQA'
            elif dirname == 'ocr_vqa':
                dirname = 'OCR-VQA'
            else:
                raise ValueError
            return os.path.join(self.image_folder, dirname, image_fn)
        else:
            return None
    
    def get_image_fn(self, i):
        item = self.meta[i]
        if 'image' in item:
            image_fn = item['image']
            dirname = image_fn.split('/')[0]
            image_fn = '/'.join(image_fn.split('/')[1:])
            all_sub_images = []
            all_sub_image_masks = []
            if dirname == 'vg':
                dirname = 'VG'
            elif dirname == 'coco':
                dirname = 'COCO2017'
            elif dirname == 'textvqa':
                dirname = 'TextVQA'
            elif dirname == 'gqa':
                dirname = 'GQA'
            elif dirname == 'ocr_vqa':
                dirname = 'OCR-VQA'
            else:
                raise ValueError

            return os.path.join(self.image_folder, dirname, image_fn)
        else:
            return None
    
    def __getitem__(self, i):
        assert self.output_mode == 'conversation'
        
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)
        
        item = self.meta[i]
        if 'image' in item:
            image_fn = item['image']
            dirname = image_fn.split('/')[0]
            image_fn = '/'.join(image_fn.split('/')[1:])
            all_sub_images = []
            all_sub_image_masks = []
            if dirname == 'vg':
                dirname = 'VG'
            elif dirname == 'coco':
                dirname = 'COCO2017'
            elif dirname == 'textvqa':
                dirname = 'TextVQA'
            elif dirname == 'gqa':
                dirname = 'GQA'
            elif dirname == 'ocr_vqa':
                dirname = 'OCR-VQA'
            else:
                raise ValueError

            tgt_img = [Image.open(os.path.join(self.image_folder, dirname, image_fn)).convert('RGB')] 
            image_label_masks = [0]

            new_conversation = copy.deepcopy(item['conversations'])
            all_box_info = [[0.0, 0.0, 1.0, 1.0]]
            all_sub_images = []
            for msg in new_conversation:
                # msg['value'] = msg['value'].replace('<image>', ALL_IMG_TOKENS_STR)
                new_value, sub_image, box_info = self.detect_boxes(msg['value'], tgt_img[0])
                msg['value'] = new_value
                all_sub_images.extend(sub_image)
                all_box_info.extend(box_info)

            if len(all_box_info) > 1:
                # detection involved
                for msg in new_conversation:
                    msg['value'] = msg['value'].replace('<image>', ALL_IMG_TOKENS_STR+DEFAULT_GRD_TOKEN)
            else:
                for msg in new_conversation:
                    msg['value'] = msg['value'].replace('<image>', ALL_IMG_TOKENS_STR)
            # if dirname != 'VG' or self.no_process:
            #     for message in item['conversations']:
            #         message['value'] = message['value'].replace('<image>', ALL_IMG_TOKENS_STR)
            #     new_conversation = item['conversations']
            #     all_box_info = [[0.0, 0.0, 1.0, 1.0]]
            # else:
            #     # need further processing here
            #     new_conversation, all_sub_images, all_sub_image_masks, all_box_info = self.process_vg(item['conversations'], tgt_img[0])
            #     for message in new_conversation:
            #         message['value'] = message['value'].replace('<image>', ALL_IMG_TOKENS_STR+DEFAULT_GRD_TOKEN)
            if self.object_format == 'image':
                return {'input_images': tgt_img + all_sub_images, 'conversation': new_conversation, 'image_label_masks': image_label_masks + all_sub_image_masks}
            else:
                return {'input_images': tgt_img, 'conversation': new_conversation, 'image_label_masks': image_label_masks + all_sub_image_masks, 'box': all_box_info}
        else:
            tgt_img = None
            image_label_masks = None
            return {'conversation': item['conversations']}
        