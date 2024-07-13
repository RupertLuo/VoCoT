from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator

import os
import re
import random
import json, pickle

import torch
from torch.utils.data import Dataset

from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from PIL import Image
from argparse import ArgumentParser

from ..utils.template import en_template_task
from ..utils.action_utils import extract_gt_action
from ..utils.data_utils import get_image_transform, padding_to_target
from ..utils.data_utils import IMG_RES_TEMPLATES, resolution_match


IGNORE_TOKEN_ID = LabelSmoother.ignore_index # -100


@dataclass
class AITWCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = 'longest'
    max_length: Optional[int] = 2048
    ignore_token_id: int = IGNORE_TOKEN_ID
    return_tensors: str = "pt"

    def __call__(self, features: List[dict], return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        batch = {
            'input_ids': [], 
            'attention_mask': [],
            'labels': []
        }

        max_input_len = max([len(ex['input_ids']) for ex in features])
        max_output_len = max([len(ex['labels']) for ex in features])
        max_length = min(self.max_length, max_input_len + max_output_len + 1)
        for example in features:
            input_ids = example['input_ids'] + example['labels'] + [self.tokenizer.pad_token_id]
            attention_mask = [1] * len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * (max_length - len(input_ids))
            attention_mask += [0] * (max_length - len(attention_mask))
            batch['input_ids'].append(input_ids[:max_length])
            batch['attention_mask'].append(attention_mask[:max_length])
            
            remainder = [self.ignore_token_id] * len(example['input_ids'])
            labels = remainder + example['labels'] + [self.tokenizer.pad_token_id]
            labels += [self.ignore_token_id] * (max_length - len(labels))
            batch['labels'].append(labels[:max_length])
        
        batch['input_ids'] = torch.tensor(batch['input_ids'], dtype=torch.int)
        batch['attention_mask'] = torch.tensor(batch['attention_mask'], dtype=torch.bool)
        batch['labels'] = torch.tensor(batch['labels'], dtype=torch.long)
        batch['images'] = torch.stack([ex['images'] for ex in features], dim=0) # B,C,H,W
    
        return batch


class AITWDataset(Dataset):
    """
    Creating a custom dataset for reading the processed AITW data
    with GPT-4V labeled detail semantic annotations.
    """
    DATASET_DIR = {
        'general': '{}/general',
        'google_apps': '{}/google_apps',
        'install': '{}/install',
        'single': '{}/single',
        'web_shopping': '{}/web_shopping',
    }

    def __init__(self,
                 data_dir: str,
                 tokenizer: PreTrainedTokenizer,
                 args: ArgumentParser,
                 mode: Optional[str]="normal"
    ):
        """
        Initialize a Dataset class

        Args:
            data_dir (str): path to pickled aitw episodes
            tokenizer (transformers.tokenizer): tokenize sentences
            args: args used here, must include attibutes
                --max_input_length
                --max_input_length
                --use_image_path
                --use_dynamic_resolution
                --use_history_action
                --use_history_image
                --max_history_length
                --use_post_think
        """
        super().__init__()

        self.data_dir = data_dir
        self.tokenizer = tokenizer

        assert mode in ['easy', 'normal', 'hard'], \
            "Choose the data process mode from ['easy', 'normal', 'hard']!"
        self.mode = mode

        self.max_src_length = args.max_input_length
        self.max_tgt_length = args.max_input_length
        
        # data options
        self.use_image_path = args.use_image_path
        self.use_dynamic_res = args.use_dynamic_resolution
        self.image_transform = {
            res_temp: get_image_transform(res_temp)
            for res_temp in IMG_RES_TEMPLATES
        }
        self.use_ground_ops = getattr(args, 'use_ground_operation', False)

        # prompt options
        self.use_history_action = getattr(args, 'use_history_action', False)
        self.use_history_image = getattr(args, 'use_history_image', False)
        self.history_len = getattr(args, 'max_history_length', 1)
        self.use_screen_desc = getattr(args, 'use_screen_desc', False)
        self.use_post_think = getattr(args, 'use_post_think', False)
        self.use_pre_think = getattr(args, 'use_pre_think', False)
        self.use_action_desc = getattr(args, 'use_action_desc', False)
        self.use_task_prompt = getattr(args, 'use_task_prompt', False)

        self.data = self._load_data()
        random.shuffle(self.data)
    
    def _load_data(self): 
        data = []
        for subset in self.DATASET_DIR:
            subdata_dir = self.DATASET_DIR[subset].format(self.data_dir)
            if os.path.exists(subdata_dir):
                filenames = [fn.replace('.pkl', '') for fn in \
                             os.listdir(subdata_dir) if fn.endswith('.pkl')]
                tmp_list = [subset] * len(filenames)
                for fn in filenames:
                    json_path = os.path.join(subdata_dir, fn, f"{fn}.json")
                    if os.path.exists(json_path):
                        episode = json.load(open(json_path, 'r'))
                    else:
                        pkl_path = os.path.join(subdata_dir, f"{fn}.pkl")
                        episode = pickle.load(open(pkl_path, 'rb'))
                    step_ids, input_texts, target_texts, images, actions = self.pre_process(episode)
                    data.extend([*zip(tmp_list, step_ids, input_texts, target_texts, images, actions)])
        return data
    
    def __len__(self, ): 
        """ Returns the length of dataset """
        return len(self.data)

    def __getitem__(self, index):
        """ return the input ids, images, attention masks and target ids """
        subset, step_id, input_text, target_text, image_list, action_tuple = self.data[index]

        if self.use_image_path:
            images = []
            for img_path in image_list:
                img, img_res = self._transform(
                    Image.open(os.path.join(self.data_dir, img_path)))
                images.append(img)
            image_list = images
        else:
            last_img, img_res = image_list[-1]
            image_list = image_list[:-1] + [last_img]

        if not self.use_history_image:
            assert len(image_list) == 1, \
                "You should only use one image when the history is forbidden!"
        
        input_ = self.tokenizer(input_text, 
                                truncation=True, 
                                max_length=self.max_src_length,
                                image_height=img_res[0],
                                image_width=img_res[1],)
        input_ids = input_.input_ids

        target_ = self.tokenizer(target_text,
                                 truncation=True,
                                 max_length=self.max_tgt_length)
        target_ids = target_.input_ids

        # if index <= 10:
        #     print('[INPUT]\n',  input_text, "\n", len(input_ids))
        #     print('[TARGET]\n', target_text, "\n", len(target_ids))
        #     import sys
        #     sys.stdout.flush()

        (action_desc, action_type, action_ground) = action_tuple
        return {
            'step_id': f'{subset.upper()}_{step_id}',
            # next 3 keys are used for training
            'input_ids': input_ids,
            'images': image_list[0],
            'labels': target_ids,
            # following 3 keys are used for evaluation
            'action_desc': action_desc,
            'action_type': action_type,
            'action_ground': action_ground,
            # following 2 keys are for debugging
            'input_text': input_text,
            'target_text': target_text,
        }

    def _transform(self, image:Image, 
                   padding:Optional[bool] = False, 
                   mask_padding: Optional[bool] = False,
        ):
        image = image.convert('RGB')
        (w, h) = image.size

        if self.use_dynamic_res:
            res_template, _pad, _pad_size_hw = resolution_match(image)
        else:
            res_template, _pad = (896, 896), padding
            if _pad: _pad_size_hw = (max(w, h), max(w, h))
        
        if padding is None: padding = _pad
        if padding: 
            image = padding_to_target(image, _pad_size_hw, mode="upleft")
            (new_w, new_h) = image.size
        
        image_tensor = self.image_transform[res_template](image)    # C, H, W
        if padding and mask_padding:
            image_mask = torch.zeros((1, new_h, new_w))             # 1, H, W
            image_mask[0, :h, :w] = 1
            resize_transform = self.image_transform[res_template].transforms[0]
            image_mask = resize_transform(image_mask).bool()
        else: 
            image_mask = 1

        return image_tensor * image_mask, res_template

    def pre_process(self, episode):
        if self.mode == "normal":
            return self._normal_preprocess_(episode)
        elif self.mode == "easy":
            raise NotImplementedError
            return self._easy_preprocess_(episode)
        elif self.mode == "hard":
            return self._hard_preprocess_(episode)
        elif self.mode == "screen":
            raise NotImplementedError
            return self._caption_preprocess_(episode)
        else:
            raise NotImplementedError
    
    def _get_query(self, instr, use_template=False, prob=0.3):
        if not use_template or random.uniform(0, 1) < 1-prob:
            instr = instr[0].upper() + instr[1:]
            if instr[-1].isalnum(): return instr + "."
            elif instr[-1] in ['"', "'"]: return instr + "."
            else: return instr
        else:
            if instr[0].isupper():
                instr = instr[0].lower() + instr[1:]
            if instr[-1] in ['.', '!', '?', ',', ';', ]:
                instr = instr[:-1]
            question = random.choice(en_template_task).format(instr)
            return question

    def _normal_preprocess_(self, episode):
        """ screen + screen_desc -> pre_action_think + action_plan """
        step_ids, input_texts, target_texts, images, action_tuples = [], [], [], [], []

        history_actions, history_images = [], []
        for ex_idx, example in enumerate(episode):
            episode_id, step_id = example['episode_id'], example['step_id']
            cur_step = f"{episode_id}_STEP-{step_id}"
            step_ids.append(cur_step)

            if not self.use_image_path and 'image' in example:
                image, image_res = self._transform(Image.fromarray(example['image']))
                cur_images = [(image, image_res)]
            else: 
                self.use_image_path = True
                assert 'image_path' in example, "No image path found!"
                image = example['image_path']
                cur_images = [image]

            instr = example['instruction'].strip()
            query = f"Query: {self._get_query(instr)}"

            prev_actions = ""
            if self.use_history_action and history_actions:
                prev_actions = ", then ".join(history_actions[-self.history_len:])
                prev_actions = f"History Action: {prev_actions}."
                if self.use_post_think:
                    prev_post_think = episode[ex_idx-1]['gpt_post_think'] \
                        if 'gpt_post_think' in example else episode[ex_idx-1]['post_think']
                    prev_actions = f"{prev_actions} {prev_post_think}"
            prev_actions = prev_actions + "\n" if prev_actions else ""
            
            if self.use_history_image and history_images:
                hist_images = history_images[-self.history_len:]
                cur_images = hist_images + cur_images
                prev_images = []
                for i in range(len(hist_images)):
                    prev_images.append(f"<img>{i}</img>")
                prev_images = " ".join(prev_images)
                prev_images = f"History Screens: {prev_images}"
                query = f"{query}\n{prev_images}"
            images.append(cur_images)
            
            screen_idx = len(cur_images) - 1
            screen_desc = ""
            if self.use_screen_desc:
                screen_desc = f"Screen Description: {example['gpt_screen_desc']}"
                screen_desc = f"Screen: <img>{screen_idx}</img>\n{screen_desc}"
            else:
                screen_desc = f"Screen: <img>{screen_idx}</img>"
            screen_desc += "\n"
            
            # TODO: Change here
            task_prompt = "TASK: Given above information, " + \
                "think about your next action to fulfill the query. " + \
                "You can perform the following 5 types of actions: " + \
                "1. SCROLL: this action moves the current page to a desired direction. " + \
                "2. CLICK: click or long press a UI element on the screen. " + \
                "3. PRESS: this action includes three system-level opertations, " + \
                    "(1) press the home button, return to the home screen; " + \
                    "(2) press the back button, return to the previous interface; " + \
                    "(3) press enter to confirm the text input and start search. " + \
                "4. TYPE: this action inserts text input to an input field/box/bar. " + \
                "5. STOP: stop and mark the query as completed or failed."
            if not self.use_task_prompt:
                input_txt = f"{screen_desc}{prev_actions}{query}"
            else:
                input_txt = f"{query}\n{screen_desc}{prev_actions}\n{task_prompt}"
            input_texts.append(input_txt)

            # check result action dual points
            for reskey in ['result_lift_yx', 'result_touch_yx', 'image_size', 'result_bbox']:
                if reskey in example and type(example[reskey]) == str:
                    example[reskey] = json.loads(example[reskey])

            action_cot = example['gpt_pre_think']
            action_cot = action_cot if action_cot[-1] == '.' else action_cot + "."
            action_desc, action_type, action_ground = \
                extract_gt_action(example, cord='rel', use_xy=False)
            if 'human_act_ground' in example: 
                action_desc = example['human_act_ground']
            elif action_type == "CLICK":
                action_desc = example['gpt_act_ground'][0]
            action_desc = action_desc if action_desc[-1] != '.' else action_desc[:-1]
            action_tuples.append((action_desc, action_type, action_ground))

            target_txt = []
            if self.use_pre_think:
                target_txt.append(f'[Think] {action_cot}')
            if self.use_action_desc:
                target_txt.append(f'[Next Action] {action_desc}.')
            if self.use_ground_ops: 
                target_txt.append(f'[Grounded Operation] action_type: {action_type}, {action_ground}.')
            target_txt = '\n'.join(target_txt)
            target_texts.append(target_txt)

            if self.use_history_action: history_actions.append(action_desc)
            if self.use_history_image: history_images.append(image)

        return step_ids, input_texts, target_texts, images, action_tuples



    def _hard_preprocess_(self, episode):
        """ screen + screen_desc -> pre_action_think + action_plan """
        step_ids, input_texts, target_texts, images, action_tuples = [], [], [], [], []

        history_actions, history_images = [], []
        for ex_idx, example in enumerate(episode):
            episode_id, step_id = example['episode_id'], example['step_id']
            cur_step = f"{episode_id}_STEP-{step_id}"
            step_ids.append(cur_step)

            if not self.use_image_path and 'image' in example:
                image, image_res = self._transform(Image.fromarray(example['image']))
                cur_images = [(image, image_res)]
            else: 
                self.use_image_path = True
                assert 'image_path' in example, "No image path found!"
                image = example['image_path']
                cur_images = [image]

            instr = example['instruction'].strip()
            query = f"Query: {self._get_query(instr)}"

            prev_actions = ""
            if self.use_history_action and history_actions:
                prev_actions = ", then ".join(history_actions[-self.history_len:])
                prev_actions = f"History Action: {prev_actions}."
                if self.use_post_think:
                    prev_post_think = episode[ex_idx-1]['gpt_post_think'] \
                        if 'gpt_post_think' in example else episode[ex_idx-1]['post_think']
                    prev_actions = f"{prev_actions} {prev_post_think}"
            prev_actions = prev_actions + "\n" if prev_actions else ""
            
            if self.use_history_image and history_images:
                hist_images = history_images[-self.history_len:]
                cur_images = hist_images + cur_images
                prev_images = []
                for i in range(len(hist_images)):
                    prev_images.append(f"<img>{i}</img>")
                prev_images = " ".join(prev_images)
                prev_images = f"History Screens: {prev_images}"
                query = f"{query}\n{prev_images}"
            images.append(cur_images)
            
            screen_idx = len(cur_images) - 1
            # screen_desc = ""
            # if self.use_screen_desc:
            #     screen_desc = f"Screen Description: {example['gpt_screen_desc']}"
            #     screen_desc = f"Screen: <img>{screen_idx}</img>\n{screen_desc}"
            # else:
            #     screen_desc = f"Screen: <img>{screen_idx}</img>"
            # screen_desc += "\n"
            screen_txt = f"Screen: <img>{screen_idx}</img>\n"
            
            # TODO: Change here
            task_prompt = "TASK: Given above information, " + \
                "think about your next action to fulfill the query. " + \
                "You can perform the following 5 types of actions: " + \
                "1. SCROLL: this action moves the current page to a desired direction. " + \
                "2. CLICK: click or long press a UI element on the screen. " + \
                "3. PRESS: this action includes three system-level opertations, " + \
                    "(1) press the home button, return to the home screen; " + \
                    "(2) press the back button, return to the previous interface; " + \
                    "(3) press enter to confirm the text input and start search. " + \
                "4. TYPE: this action inserts text input to an input field/box/bar. " + \
                "5. STOP: stop and mark the query as completed or failed."
            if not self.use_task_prompt:
                input_txt = f"{screen_txt}{prev_actions}{query}"
            else:
                input_txt = f"{query}\n{screen_txt}{prev_actions}\n{task_prompt}"
            input_texts.append(input_txt)

            # check result action dual points
            for reskey in ['result_lift_yx', 'result_touch_yx', 'image_size', 'result_bbox']:
                if reskey in example and type(example[reskey]) == str:
                    example[reskey] = json.loads(example[reskey])
            
            screen_desc = example['gpt_screen_desc']
            action_cot = example['gpt_pre_think']
            action_cot = action_cot if action_cot[-1] == '.' else action_cot + "."
            action_desc, action_type, action_ground = \
                extract_gt_action(example, cord='rel', use_xy=False)
            if 'human_act_ground' in example: 
                action_desc = example['human_act_ground']
            elif action_type == "CLICK":
                action_desc = example['gpt_act_ground'][0]
            action_desc = action_desc if action_desc[-1] != '.' else action_desc[:-1]
            action_tuples.append((action_desc, action_type, action_ground))

            target_txt = f'[Observe] {screen_desc}\n[Think] {action_cot}\n[Next Action] {action_desc}.'
            if self.use_ground_ops: target_txt += \
                         f'\n[Grounded Operation] action_type: {action_type}, {action_ground}.'
            target_texts.append(target_txt)

            if self.use_history_action: history_actions.append(action_desc)
            if self.use_history_image: history_images.append(image)

        return step_ids, input_texts, target_texts, images, action_tuples

