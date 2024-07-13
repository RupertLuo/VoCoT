from torch.utils.data import Dataset
import json
import os
import random
import copy
from PIL import Image
from constants import *
from ..utils.box_utils import box2str, reshape_box
import math

class MP3DDataset(Dataset):
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
        self.meta = json.load(open(path))
        self.binary_templates = [
            'Find the {obj1}{box1} and the {obj2}{box2}, based on the visual information, the answer is {answer}',
            'Based on the query, first I find the {obj1}{box1}, then locate the {obj2}{box2}, according to their positions and relationship, {answer}',
            'To answer the question, first extract the target objects, in the image, there is the {obj1}{box1} and the {obj2}{box2}, the conclusion is that {answer}',
            'Find the {obj1}{box1}, locate the {obj2}{box2}, according to the visual information of these objects, {answer}',
            'To distinguish the spatial relationship, we need to detect the objects in the question, get {obj1}{box1} and {obj2}{box2}, check their spatial relationship, the conclusion is {answer}',
            'By searching in the image, I can find {obj1}{box1} and {obj2}{box2}, referring to the extracted information, the answer is {answer}'
        ]
        self.far_templates = [
            'Based on the image, I find all candidate objects: {objects}, among which the {answer} is the farthest.',
            'By checking the location of objects in your question, locate {objects}, the {answer} is the farthest one within the image.',
            'Comparing the positions of {objects}, the answer is the {answer}.',
            'To find the farthest object among them, I detect the related objects: {objects}, referring to the extracted information, the answer is the {answer}.'
        ]
        self.close_templates = [
            'Based on the image, I find all candidate objects: {objects}, among which the {answer} is the closest.',
            'By checking the location of objects in your question, locate {objects}, the {answer} is the nearest one within the image.',
            'Comparing the positions of {objects}, the answer is the {answer}.',
            'To find the closest object among them, I detect the related objects: {objects}, referring to the extracted information, the answer is the {answer}.'
        ]
        print('MP3D-Spatial has {} samples'.format(len(self)))

    def proc_box(self, box, image):
        w, h = image.size
        new_box = [c for c in box]
        new_box[2] = new_box[2] + new_box[0]
        new_box[3] = new_box[3] + new_box[1]
        new_box = [c / w if (i%2==0) else c / h for i,c in enumerate(new_box)]
        if self.expand2square:
            new_box = reshape_box(image, new_box)
        return new_box

    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, index):
        item = self.meta[index]
        image = os.path.split(item['image'])[-1]
        assert self.object_format == 'representation'
        image = Image.open(os.path.join(self.image_path, image)).convert('RGB')
        conversation = []

        # all box information
        all_box_input = [[0.0, 0.0, 1.0, 1.0]]
        image_label_masks = [0]
        sub_image = []

        # process the input and output
        input_query = random.choice(item['questions'])
        refined_query = ALL_IMG_TOKENS_STR + DEFAULT_GRD_TOKEN + '\n'

        if item['relation'] not in ['close', 'far']:
            prompt_template = random.choice(self.binary_templates)
            refined_query = refined_query + input_query + ' ' + COT_ACTIVATION

            answer = item['answer_options'][item['answer']].lower()
            refined_response = ''
            # objects
            obj1, box1 = item['obj1']
            obj2, box2 = item['obj2']
            box1 = self.proc_box(box1, image)
            box2 = self.proc_box(box2, image)
            all_box_input.extend([box1, box2])
            box1_str = DEFAULT_BOC_TOKEN + box2str(box1, mode=self.phrase_format, prec=self.phrase_prec, space=self.phrase_space) + DEFAULT_EOC_TOKEN + ALL_IMG_TOKENS_STR
            box2_str = DEFAULT_BOC_TOKEN + box2str(box2, mode=self.phrase_format, prec=self.phrase_prec, space=self.phrase_space) + DEFAULT_EOC_TOKEN + ALL_IMG_TOKENS_STR
            refined_response = prompt_template.format(
                obj1=obj1, box1=box1_str, obj2=obj2, box2=box2_str, answer=answer
            )
        else:
            refined_query = refined_query + input_query + ' Select from the following options: {}.'.format('; '.join(item['answer_options'])) + ' ' + COT_ACTIVATION
            prompt_template = random.choice(self.far_templates) if item['relation'] == 'far' else random.choice(self.close_templates)
            answer = item['answer_options'][item['answer']]
            refined_response = ''
            # objects
            all_objects_str = []
            for i, obj in enumerate(item['answer_options']):
                box = item['obj_bboxes'][i]
                box = self.proc_box(box, image)
                all_box_input.append(box)
                box_str = DEFAULT_BOC_TOKEN + box2str(box, mode=self.phrase_format, prec=self.phrase_prec, space=self.phrase_space) + DEFAULT_EOC_TOKEN + ALL_IMG_TOKENS_STR
                obj_str = obj + box_str
                all_objects_str.append(obj_str)
            refined_response = prompt_template.format(
                objects=', '.join(all_objects_str), answer=answer
            )


        conversation = [
            {'from': 'human', 'value': refined_query},
            {'from': 'gpt', 'value': refined_response}
        ]

        # if need further include short answer
        # only for GQA-COT and GPT4GEN_BoxCoT
        if self.further_instruct:
            conversation.extend([
                {'from': 'human', 'value': 'What is your final answer?'},
                {'from': 'gpt', 'value': item['relation']}]
            )
        if self.object_format == 'image':
            return {'input_images': [image, sub_image] if not self.ignore_object else [image], 'conversation': conversation, 'image_label_masks': image_label_masks}
        elif self.object_format == 'representation':
            return {'input_images': [image] , 'conversation': conversation, 'image_label_masks': image_label_masks, 'box': all_box_input}
        else:
            raise ValueError