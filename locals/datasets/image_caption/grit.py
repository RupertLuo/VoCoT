from email.policy import default
from torch.utils.data import Dataset
import json as js
import math
import random
import os
from PIL import Image
from collections import defaultdict
from constants import *
from ..utils.box_utils import box2str, reshape_box, reshape_box_reverse

class GriTDataset(Dataset):

    def __init__(self, 
                 path: str,
                 image_folder: str,
                 meta_folder = str,
                 instruct: bool = False,
                 min_resize_res: int = 256,
                 max_resize_res: int = 256,
                 crop_res: int = 256,
                 flip_prob: float = 0.5,
                 sample_weight: float = 1.0,
                 check: bool = False,
                 output_mode: str = 'text',
                 raw_image: bool = False,
                 shuffle: bool=False,
                 phrase_key: str='noun_chunks',
                 avoid_image_gen: bool=False,
                 phrase_format: str='special_tokens',
                 phrase_prec: int=2,
                 expand2square: bool=False,
                 object_format: str='image',
                 **kwargs):
        # load from json ../datasets/gqa-inpaint/meta_info.json
        self.path = path
        self.instruct = instruct
        self.meta_folder = meta_folder
        self.phrase_key = phrase_key
        self.meta = []
        if path.endswith('txt'):
            with open(path) as rf:
                for line in rf:
                    self.meta.append(line.strip())
            self.need_reload = True
        else:
            self.need_reload = False
            self.meta = js.load(open(path))
        self.image_folder = image_folder
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.check = check
        self.raw_image = raw_image
        self.object_format = object_format
        self.output_mode = output_mode
        self.avoid_image_gen = avoid_image_gen
        self.phrase_format = phrase_format
        self.phrase_prec = phrase_prec
        self.shuffle = shuffle # shuffle is for interleaved image-text data
        self.expand2square = expand2square

        self.flip_prob = flip_prob
        self.sample_weight = sample_weight
        print(f"GRIT has {len(self)} samples!!")

    def __len__(self):
        return int(len(self.meta) * self.sample_weight)

    def merge_chunks(self, phrases, tgt_img):
        chunk_dict = defaultdict(list)
        for p in phrases:
            tmp_box = [k for k in p[2:-1]]
            if self.expand2square:
                tmp_box = reshape_box(tgt_img, tmp_box)
            chunk_dict[tuple(p[:2])].append(tmp_box)
        return [(k,v) for k,v in chunk_dict.items()]
    
    def proc_chunks(self, phrases, cap_len, tgt_img):
        phrases = self.merge_chunks(phrases, tgt_img)
        phrases = sorted(phrases, key = lambda x: x[0][0])
        all_phrases = []
        current_index = 0
        for p in phrases:
            index, boxes = p
            start, end = [int(c) for c in index]
            if start > current_index:
                all_phrases.append([current_index, start, False, None])
                all_phrases.append([start, end, True, boxes])
                current_index = end
            elif start == current_index:
                all_phrases.append([start, end, True, boxes])
                current_index = end
            else:
                raise ValueError
        if current_index < cap_len:
            all_phrases.append([current_index, cap_len, False, None])
        return all_phrases
    
    def process_phrase(self, text, bounding_boxes, image):
        # crop the image
        width, height = image.size
        phrase_format = DEFAULT_BOP_TOKEN + text + DEFAULT_EOP_TOKEN
        sub_images = []
        sub_texts = []
        for box in bounding_boxes:
            if self.expand2square:
                original_box = reshape_box_reverse(image, box)
            x_min, y_min, x_max, y_max = original_box
            x_min = x_min*width
            y_min = y_min*height
            x_max = x_max*width
            y_max = y_max*height
            sub_image = image.crop((x_min, y_min, x_max, y_max))
            sub_images.append(sub_image)
            # coor = [round(c*(LOC_TOKEN_NUM-1)) for c in box]
            # coor = ''.join([ALL_LOC_TOKENS[i] for i in coor])
            coor = box2str(box, mode=self.phrase_format, prec=self.phrase_prec)
            coor_format = DEFAULT_BOC_TOKEN + coor + DEFAULT_EOC_TOKEN
            sub_texts.append(coor_format + ALL_IMG_TOKENS_STR if self.object_format != 'coordinate' else coor_format)
        # <p>phrase</p><coor>[LOC]</coor><Img><ImageHere></Img>
        return phrase_format + DEFAULT_SEP_TOKEN.join(sub_texts), sub_images

    def process_phrase_loc(self, bounding_boxes, width, height):
        # crop the image
        sub_texts = []
        for box in bounding_boxes:
            x_min, y_min, x_max, y_max = box
            x_min = x_min*width
            y_min = y_min*height
            x_max = x_max*width
            y_max = y_max*height
            # coor = [round(c*(LOC_TOKEN_NUM-1)) for c in box]
            # coor = ''.join([ALL_LOC_TOKENS[i] for i in coor])
            coor = box2str(box, mode=self.phrase_format, prec=self.phrase_prec)
            sub_texts.append(coor)
        # <p>phrase</p><coor>[LOC]</coor><Img><ImageHere></Img>
        return sub_texts
    
    def get_image_fn(self, image_id):
        return os.path.join(self.image_folder, '{}.jpg'.format(image_id))
    
    def __getitem__(self, i):
        
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)

        if self.need_reload:
            meta_fn = self.meta[i]
            item = js.load(open(os.path.join(self.meta_folder, meta_fn)))
        else:
            item = self.meta[i]
        image_id = item['key']
        tgt_img = Image.open(self.get_image_fn(image_id)).convert('RGB')
        caption = item['caption']
        cap_len = len(caption)

        # process the phrases
        phrases = item[self.phrase_key]
        phrases = self.proc_chunks(phrases, cap_len, tgt_img)

        # start from <Img><ImageHere></Img><grounding>
        instruction = ALL_IMG_TOKENS_STR+DEFAULT_GRD_TOKEN
        input_images = [tgt_img]
        image_label_masks = [0]
        all_box_info = [[0.0, 0.0, 1.0, 1.0]]
        for p in phrases:
            start, end, ground, box_info = p
            if ground:
                txt, sub_image = self.process_phrase(caption[start:end], box_info, tgt_img)
                all_box_info.extend(box_info)
                instruction = instruction + txt
                input_images.extend(sub_image)
                if self.avoid_image_gen:
                    image_label_masks.extend([0]*len(sub_image))
                else:
                    image_label_masks.extend([1]*len(sub_image))
            else:
                instruction = instruction + caption[start:end]

        if self.output_mode == 'conversation':
            raise ValueError
        elif self.output_mode == 'text':
            # print(instruction)
            if self.object_format == 'image':
                return {'input_images': input_images, 'text': instruction, 'image_label_masks': image_label_masks}
            elif self.object_format == 'representation':
                return {'input_images': input_images[:1], 
                        'text': instruction, 
                        'image_label_masks': image_label_masks,
                        'box': all_box_info}
            elif self.object_format == 'coordinate':
                return {'input_images': input_images[:1], 
                        'text': instruction, 
                        'image_label_masks': image_label_masks,
                        'box': all_box_info[:1]}
            else:
                raise ValueError
    
    def get_tmp(self, i):
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)

        if self.need_reload:
            meta_fn = self.meta[i]
            item = js.load(open(os.path.join(self.meta_folder, meta_fn)))
        else:
            item = self.meta[i]
        image_id = item['key']
        caption = item['caption']
        cap_len = len(caption)

        # process the phrases
        phrases = item[self.phrase_key]
        width = item['width']
        height = item['height']
        phrases = self.proc_chunks(phrases, cap_len)
        all_locs = []
        for p in phrases:
            start, end, ground, box_info = p
            if ground:
                texts = self.process_phrase_loc(box_info, width, height)
                all_locs.extend(texts)
            else:
                continue
        return all_locs

        
