from typing import Any, Optional, Dict, List

import torch
from transformers import PreTrainedTokenizer

from constants import *
from PIL import Image
import copy
from conversation import CONV_VISION, CONV_VISION_MISTRAL
import conversation as conversation_lib
import transformers
import re


def tokenizer_image_token(prompt, tokenizer, image_token_index=-200, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def _tokenize_fn(strings,
                 tokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def _mask_targets(target, tokenized_lens, speakers, only_mask_system = False):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    if not only_mask_system:
        for tokenized_len, speaker in zip(tokenized_lens, speakers):
            if speaker == "human":
                target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
            cur_idx += tokenized_len

def tokenizer_image_token(prompt, tokenizer, image_token_index=-200, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def preprocess_llama_2(
    tmp_sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    inference: bool = False,
) -> Dict:
    conv = CONV_VISION_MISTRAL.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    if isinstance(tmp_sources[0], dict):
        sources = [tmp_sources]
    else:
        sources = tmp_sources
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        if inference:
            conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len and not inference:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids.squeeze(),
        labels=targets.squeeze(),
    )

def preprocess_v1(
        source,
        tokenizer: transformers.PreTrainedTokenizer,
        has_image: bool = False,
        inference: bool = False,
        only_mask_system: bool = False,
) -> Dict:
    '''
    FIXME: support only_mask_system=True
    '''
    conv = CONV_VISION.copy()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    if roles[source[0]["from"]] != conv.roles[0]:
        # Skip the first one if it is not from human
        source = source[1:]

    conv.messages = []
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2], f"{i}"
        conv.append_message(role, sentence["value"])
    if inference:
        conv.append_message(conv.roles[1], None)

    conversation = conv.get_prompt()

    # Mask targets
    rounds = conversation.split(conv.sep2)

    input_ids_ = torch.tensor([1], dtype=torch.int64)
    targets_ = torch.tensor([-100], dtype=torch.int64)
    for i, rou in enumerate(rounds):
        if rou == "":
            continue
        if (not inference) or  (i < (len(rounds) - 1)):
            rou += conv.sep2
        if has_image:
            cur_input_ids_ = tokenizer_image_token(rou, tokenizer, return_tensors='pt')[1:]
            input_ids_ = torch.cat([input_ids_, cur_input_ids_], dim=0)
            if only_mask_system:
                mask_len = len(tokenizer_image_token(re.sub(f'{conv.roles[0]}:[\s\S]*', f'{conv.roles[0]}:', rou), tokenizer)[1:])
            else:
                mask_len = len(tokenizer_image_token(re.sub(f'{conv.roles[1]}:[\s\S]*', f'{conv.roles[1]}:', rou), tokenizer)[1:])
            # targets_ = torch.cat([targets_, torch.tensor([-100] * mask_len), cur_input_ids_[mask_len:]], dim=0)
            targets_ = torch.cat([targets_, torch.tensor([-100] * mask_len), cur_input_ids_[mask_len:]], dim=0)
        else:
            cur_input_ids_ = tokenizer(rou, return_tensors='pt')["input_ids"][0, 1:]
            input_ids_ = torch.cat([input_ids_, cur_input_ids_], dim=0)
            mask_len = len(tokenizer(re.sub(f'{conv.roles[1]}:[\s\S]*', f'{conv.roles[1]}:', rou))["input_ids"][1:])
            # targets_ = torch.cat([targets_, torch.tensor([-100] * mask_len), cur_input_ids_[mask_len:]], dim=0)
            targets_ = torch.cat([targets_, torch.tensor([-100] * mask_len), cur_input_ids_[mask_len:]], dim=0)
    return {"input_ids": input_ids_, "labels": targets_}

def _add_speaker_and_signal(header, source, conv, get_conversation=True, inference = False):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    END_SEQ_SIGNAL = '###'
    conversation = header
    for sentence in source:
        from_str = sentence["from"].strip()
        if from_str.lower() == "human":
            from_str = conv.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conv.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += END_SEQ_SIGNAL if not inference else BEGIN_SIGNAL + conv.roles[1] + ':'
    return conversation

    
class VoCoT_InputProcessor(object):

    def __init__(self, tokenizer: PreTrainedTokenizer, 
                 input_image_processor: Any, 
                 output_image_processor: Any,
                 expand2square: bool=False,
                 merge_in_out_image: bool=False,
                 use_mistral: bool=False,
                 inference: bool = False):
        self.tokenizer = tokenizer
        self.input_image_processor = input_image_processor
        if hasattr(self.input_image_processor, 'image_mean'):
            self.image_mean = self.input_image_processor.image_mean
        else:
            self.image_mean = (1, 1, 1)
        self.output_image_processor = output_image_processor
        self.expand2square = expand2square
        self.merge_in_out_img = merge_in_out_image
        self.inference = inference
        if use_mistral:
            self.process_fn = preprocess_llama_2
        else:
            self.process_fn = preprocess_v1

    def preprocess(self,
        sources,
        has_image = True,
        add_special_tokens=True,
        inference = False):
        # add end signal and concatenate together
        conv = CONV_VISION.copy()
        conversations = []

        for source in sources:
            header = f"{conv.system}\n\n"
            conversation = _add_speaker_and_signal(header, source, conv, inference = inference)
            conversations.append(conversation)
        # tokenize conversations
        def get_tokenize_len(prompts):
            return [len(tokenizer_image_token(prompt, self.tokenizer)) for prompt in prompts]

        if has_image:
            input_ids = [tokenizer_image_token(prompt, self.tokenizer, return_tensors='pt') for prompt in conversations]
        else:
            conversations_tokenized = _tokenize_fn(conversations, self.tokenizer)
            input_ids = conversations_tokenized["input_ids"]

        targets = copy.deepcopy(input_ids)
        for target, source in zip(targets, sources):
            if has_image:
                tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
            else:
                tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], self.tokenizer)["input_ids_lens"]
            speakers = [sentence["from"] for sentence in source]
            _mask_targets(target, tokenized_lens, speakers )

        return dict(input_ids=input_ids, labels=targets)

    def preprocess_text(self, text):
        text_inputs = self.tokenizer(
            text+self.tokenizer.eos_token if not self.inference else text,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        )
        # input_ids = text_inputs.input_ids
        labels = text_inputs.input_ids.clone()
        attention_mask = text_inputs.attention_mask
        return dict(
            input_ids=text_inputs.input_ids,
            labels = labels,
            attention_mask = attention_mask
        )

    
    def expand2square_fn(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
    def __call__(self, input_dict, inference = False, **kwargs) -> Any:
        output_dict = {}
        if inference:
            self.inference = inference
        if 'conversation' in input_dict:
            # the conversation like talk
            text_input = self.process_fn(input_dict['conversation'], self.tokenizer, has_image=False, inference = self.inference)
            output_dict['input_ids'] = text_input['input_ids']
            output_dict['attention_mask'] = torch.ones_like(text_input['input_ids'])
            output_dict['labels'] = text_input['labels']
        elif 'text' in input_dict:
            # the sequence like mode
            text_input = self.preprocess_text(input_dict['text'])
            output_dict['input_ids'] = text_input['input_ids'][0]
            output_dict['attention_mask'] = text_input['attention_mask'][0]
            output_dict['labels'] = text_input['labels'][0]
        elif 'conversations' in input_dict:
            text_input = self.process_fn(input_dict['conversations'], self.tokenizer, has_image=False, inference = self.inference)
            output_dict['input_ids'] = text_input['input_ids']
            output_dict['attention_mask'] = torch.ones_like(text_input['input_ids'])
            output_dict['labels'] = text_input['labels']
        elif 'input_ids' in input_dict:
            output_dict['input_ids'] = input_dict['input_ids']
            output_dict['attention_mask'] = torch.ones_like(input_dict['input_ids'])
            output_dict['labels'] = input_dict['input_ids'].clone()
        else:
            raise ValueError

        if self.merge_in_out_img:
            if 'input_images' not in input_dict and 'output_images' in input_dict:
                # text to image
                input_dict['input_images'] = input_dict['output_images']
            elif 'input_images' not in input_dict and 'output_images' not in input_dict:
                # text only
                pass
            elif 'input_images' in input_dict and 'output_images' in input_dict:
                # image editing
                input_dict['input_images'] = input_dict['input_images'] + input_dict['output_images']
        if 'input_images' in input_dict:
            all_images = []
            images = input_dict['input_images']
            if not isinstance(images, list):
                images = [images]
            # if self.merge_in_out_img:
            #     out_images = input_dict['output_images'] if isinstance(input_dict['output_images'], list) else [input_dict['output_images']]
            #     images = images + input_dict['output_images']
            num_images = len(images)
            if not self.expand2square:
                output_dict['raw_images'] = [img for img in images]
            else:
                output_dict['raw_images'] = [self.expand2square_fn(img, tuple([int(255*m) for m in self.image_mean])) for img in images]
            for i,img in enumerate(images):
                if type(img) is str:
                    img = Image.open(img).convert('RGB')
                if self.expand2square:
                    img = self.expand2square_fn(img, tuple([int(255*m) for m in self.image_mean]))
                image_output = self.input_image_processor(img)
                all_images.append(image_output)
            input_images = torch.stack(all_images, dim=0)
            output_dict['input_images'] = input_images
        else:
            output_dict['input_images'] = None
        
        if 'output_images' in input_dict:
            all_out_images = []
            out_images = input_dict['output_images']
            if not isinstance(out_images, list):
                out_images = [out_images]
            num_out_images = len(out_images)
            for img in out_images:
                if self.expand2square:
                    img = self.expand2square_fn(img, (255, 255, 255))
                image_output = self.output_image_processor(img)
                all_out_images.append(image_output)
            output_images = torch.stack(all_out_images, dim=0)
            output_dict['output_images'] = output_images

            if 'output_cond_images' in input_dict:
                # process the conditional images
                all_out_cond_images = []
                all_out_cond_img_mask = []
                out_cond_images = input_dict['output_cond_images']
                if not isinstance(out_cond_images, list):
                    out_cond_images = [out_cond_images]
                for img in out_cond_images:
                    if img is None:
                        # if no condition
                        cond_flag = 0
                        resolution = self.output_image_processor.transforms[1].size
                        image_output = torch.zeros((3,) + resolution).float()
                    else:
                        if type(img) is str:
                            img = Image.open(img)
                        if self.expand2square:
                            img = self.expand2square_fn(img, (255, 255, 255))
                        image_output = self.output_image_processor(img)
                        cond_flag = 1
                    all_out_cond_img_mask.append(cond_flag)
                    all_out_cond_images.append(image_output)
                all_out_cond_images = torch.stack(all_out_cond_images, dim=0)
                output_dict['output_cond_images'] = all_out_cond_images
                output_dict['output_cond_img_mask'] = torch.tensor(all_out_cond_img_mask)
            else:
                all_out_cond_images = []
                all_out_cond_img_mask = []
                for i in range(num_out_images):
                    all_out_cond_img_mask.append(0)
                    resolution = self.output_image_processor.transforms[1].size
                    zero_img = torch.zeros((3,) + resolution).float()
                    all_out_cond_images.append(zero_img)
                all_out_cond_images = torch.stack(all_out_cond_images, dim=0)
                output_dict['output_cond_images'] = all_out_cond_images
                output_dict['output_cond_img_mask'] = torch.tensor(all_out_cond_img_mask)
                
        else:
            if 'output_cond_images' in input_dict:
                # process the conditional images
                all_out_cond_images = []
                all_out_cond_img_mask = []
                out_cond_images = input_dict['output_cond_images']
                if not isinstance(out_cond_images, list):
                    out_cond_images = [out_cond_images]
                for img in out_cond_images:
                    if img is None:
                        # if no condition
                        cond_flag = 0
                        resolution = self.output_image_processor.transforms[1].size
                        image_output = torch.zeros((3,) + resolution).float()
                    else:
                        if type(img) is str:
                            img = Image.open(img)
                        if self.expand2square:
                            img = self.expand2square_fn(img, (255, 255, 255))
                        image_output = self.output_image_processor(img)
                        cond_flag = 1
                    all_out_cond_img_mask.append(cond_flag)
                    all_out_cond_images.append(image_output)
                all_out_cond_images = torch.stack(all_out_cond_images, dim=0)
                output_dict['output_cond_images'] = all_out_cond_images
                output_dict['output_cond_img_mask'] = torch.tensor(all_out_cond_img_mask)
            else:
                output_dict['output_cond_images'] = None
                output_dict['output_cond_img_mask'] = None
            output_dict['output_images'] = None

        if 'image_label_masks' in input_dict:
            output_dict['image_label_masks'] = torch.tensor(input_dict['image_label_masks'], dtype=torch.float32)
        else:
            output_dict['image_label_masks'] = None
        if 'box' in input_dict:
            output_dict['box'] = torch.tensor(input_dict['box'], dtype=torch.float32)
        else:
            if output_dict['input_images'] is None:
                output_dict['box'] = None
            else:
                output_dict['box'] = torch.tensor([[0.0, 0.0, 1.0, 1.0] for _ in range(num_images)], dtype=torch.float32)
        if 'options' in input_dict:
            assert self.inference
            output_dict['options'] = input_dict['options']
        return output_dict


    
class P2P_InputProcessor(object):

    def __init__(self, tokenizer: PreTrainedTokenizer, 
                 input_image_processor: Any, 
                 output_image_processor: Any,
                 for_minigpt: bool=False,
                 expand2square: bool=False):
        self.tokenizer = tokenizer
        self.input_image_processor = input_image_processor
        self.output_image_processor = output_image_processor
        self.for_minigpt = for_minigpt
        self.expand2square = expand2square
        

    def preprocess(self, text):
        # add end signal and concatenate together
        inputs = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding='max_length',
        truncation=True, return_tensors='pt')
        
        return inputs.input_ids

    def expand2square_fn(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result
    
    def __call__(self, input_dict, inference = False, **kwargs) -> Any:
        output_dict = {}
        if 'conversation' in input_dict:
            # remove the <image> mark
            text = input_dict['conversation'][0]['value'].replace(': <Img><ImageHere></Img>.', '')
            output_dict['input_ids'] = self.preprocess(text)
        else:
            raise ValueError
        
        if 'output_images' in input_dict:
            all_out_images = []
            out_images = input_dict['output_images']
            if not isinstance(out_images, list):
                out_images = [out_images]
            else:
                out_images = out_images[:1]
            num_out_images = len(out_images)
            for img in out_images:
                if self.expand2square:
                    img = self.expand2square_fn(img, (255, 255, 255))
                image_output = self.output_image_processor(img)
                all_out_images.append(image_output)
            output_images = torch.stack(all_out_images, dim=0)
            output_dict['output_images'] = output_images

            if 'output_cond_images' in input_dict:
                # process the conditional images
                all_out_cond_images = []
                all_out_cond_img_mask = []
                out_cond_images = input_dict['output_cond_images']
                if not isinstance(out_cond_images, list):
                    out_cond_images = [out_cond_images]
                else:
                    out_cond_images = out_cond_images[:1]
                for img in out_cond_images:
                    if img is None:
                        # if no condition
                        cond_flag = 0
                        resolution = self.output_image_processor.transforms[1].size
                        image_output = torch.zeros((3,) + resolution).float()
                    else:
                        if type(img) is str:
                            img = Image.open(img)
                        if self.expand2square:
                            img = self.expand2square_fn(img, (255, 255, 255))
                        image_output = self.output_image_processor(img)
                        cond_flag = 1
                    all_out_cond_img_mask.append(cond_flag)
                    all_out_cond_images.append(image_output)
                all_out_cond_images = torch.stack(all_out_cond_images, dim=0)
                output_dict['output_cond_images'] = all_out_cond_images
                output_dict['output_cond_img_mask'] = torch.tensor(all_out_cond_img_mask)
            else:
                all_out_cond_images = []
                all_out_cond_img_mask = []
                for i in range(num_out_images):
                    all_out_cond_img_mask.append(0)
                    resolution = self.output_image_processor.transforms[1].size
                    zero_img = torch.zeros((3,) + resolution).float()
                    all_out_cond_images.append(zero_img)
                all_out_cond_images = torch.stack(all_out_cond_images, dim=0)
                output_dict['output_cond_images'] = all_out_cond_images
                output_dict['output_cond_img_mask'] = torch.tensor(all_out_cond_img_mask)
        else:
            raise ValueError

        return output_dict