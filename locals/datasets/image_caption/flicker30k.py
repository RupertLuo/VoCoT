from email.policy import default
from torch.utils.data import Dataset
import json as js
import math
import random
import os
from PIL import Image
from collections import defaultdict
from constants import *
from ..utils.box_utils import box2str, reshape_box

class FlickrDataset(Dataset):

    def __init__(self, 
                 path: str,
                 image_folder: str,
                 instruct: bool = False,
                 sample_weight: float = 1.0,
                 check: bool = False,
                 output_mode: str = 'text',
                 raw_image: bool = False,
                 shuffle: bool=False,
                 avoid_image_gen: bool=False,
                 phrase_format: str='special_tokens',
                 phrase_prec: int=2,
                 expand2square: bool=False,
                 object_format: str='image',
                 phrase_space: bool=False,
                 **kwargs):
        # load from json ../datasets/gqa-inpaint/meta_info.json
        self.path = path
        self.instruct = instruct
        self.meta = [js.loads(line) for line in open(self.path)]
        self.image_folder = image_folder
        self.raw_image = raw_image
        self.object_format = object_format
        self.output_mode = output_mode
        self.avoid_image_gen = avoid_image_gen
        self.phrase_format = phrase_format
        self.phrase_prec = phrase_prec
        self.shuffle = shuffle # shuffle is for interleaved image-text data
        self.expand2square = expand2square
        self.phrase_space = phrase_space

        self.sample_weight = sample_weight
        print(f"Flickr30k entities has {len(self)} samples!!")

    def __len__(self):
        return int(len(self.meta) * self.sample_weight)

    def proc_box(self, box, image):
        w, h = image.size
        x_min, y_min, x_max, y_max = box
        sub_image = image.crop((x_min, y_min, x_max, y_max))
        new_box = [c / w if (i%2==0) else c / h for i,c in enumerate(box)]
        if self.expand2square:
            new_box = reshape_box(image, new_box)
        return new_box, sub_image
    
    def __len__(self):
        return len(self.meta)
    
    
    def __getitem__(self, i):
        
        if self.sample_weight >= 1:
            i = i % len(self.meta)
        else:
            remainder = math.ceil(i / self.sample_weight - int(i / self.sample_weight))
            i = int(i / self.sample_weight) + random.randint(0, int(1 / self.sample_weight) - 1 + remainder)

        item = self.meta[i]
        image_id = item['image_id']
        image = Image.open(os.path.join(self.image_folder, '{}.jpg'.format(image_id))).convert('RGB')
        caption = item['sentence']

        box_infos = []
        sub_image_infos = []
        for b in item['boxes']:
            current_reshape_infos = self.proc_box(b, image)
            box_infos.append(current_reshape_infos[0])
            sub_image_infos.append(current_reshape_infos[1])
        all_box_input = [[0.0, 0.0, 1.0, 1.0]]
        image_label_masks = [0]
        sub_image = []
        input_images = [image]

        # start from <Img><ImageHere></Img><grounding>
        input_query = item['sentence'].replace('<ph_st>', '').split('<ph_ed>')
        refined_query = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + ' '
        for box_ind, current_box in enumerate(item['boxes_seq']):
            sub_image.extend(sub_image_infos[c] for c in current_box)
            current_box = [box_infos[c] for c in current_box]
            all_box_input.extend(current_box)
            image_label_masks.extend([0]*len(current_box))
            if self.object_format == 'coordinate':
                box_in_str = DEFAULT_SEP_TOKEN.join([DEFAULT_BOC_TOKEN + box2str(c, mode=self.phrase_format, prec=self.phrase_prec, space=self.phrase_space) + DEFAULT_EOC_TOKEN for c in current_box])
            else:
                box_in_str = DEFAULT_SEP_TOKEN.join([DEFAULT_BOC_TOKEN + box2str(c, mode=self.phrase_format, prec=self.phrase_prec, space=self.phrase_space) + DEFAULT_EOC_TOKEN + ALL_IMG_TOKENS_STR for c in current_box])
            refined_query = refined_query + input_query[box_ind] + box_in_str
        refined_query = refined_query + input_query[-1]

        if self.output_mode == 'conversation':
            query = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n' + 'Briefly describe this image. Locate objects and provide bounding boxes in your response.'
            response = refined_query.replace(ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + ' ', '')
            conversation = [{'from': 'human', 'value': query}, {'from': 'gpt', 'value': response}]
            return {'input_images': input_images, 'conversation': conversation, 'image_label_masks': image_label_masks, 'box': all_box_input}
            raise ValueError
        elif self.output_mode == 'text':
            # print(instruction)
            if self.object_format == 'image':
                return {'input_images': input_images + sub_image, 'text': refined_query, 'image_label_masks': image_label_masks}
            elif self.object_format == 'representation':
                return {'input_images': input_images, 
                        'text': refined_query, 
                        'image_label_masks': image_label_masks,
                        'box': all_box_input}
            elif self.object_format == 'coordinate':
                return {'input_images': input_images, 
                        'text': refined_query, 
                        'image_label_masks': image_label_masks,
                        'box': all_box_input[:1]}
            else:
                raise ValueError