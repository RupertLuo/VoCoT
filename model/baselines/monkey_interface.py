from transformers import AutoModelForCausalLM, AutoTokenizer
from .monkey.model import MonkeyConfig, MonkeyLMHeadModel, QWenTokenizer
from dataclasses import dataclass
from typing import Dict, List
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
# from transformers import TextStreamer
# from .utils import get_image
import torch
import torch.nn as nn
from torchvision import transforms
# from utils.preprocessors import BaseProcessor, SingleChoiceProcessor, ConvSingleChoiceProcessor

class MonkeyInterface(nn.Module):
    def __init__(self, model_base=None, model_path="echo840/Monkey", device=None, half=False, inference_method='generation') -> None:
        super(MonkeyInterface, self).__init__()
        self.tokenizer = QWenTokenizer.from_pretrained(model_path, padding_side="right", use_fast=False)
        self.model = MonkeyLMHeadModel.from_pretrained(model_path, device_map='cpu', fp16=True, bf16=False).eval()
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        if half=='fp16':
            dtype = torch.bfloat16
        elif half=='bf16':
            dtype=torch.float16
        else:
            dtype = torch.float32
        self.dtype = dtype
        self.model = self.model.to(dtype).to(device)
        # self.image_processor = self.processor.image_processor
        
        # convert to the device
        self.model.to(self.device)

        # setup the inference method
        self.inference_method = inference_method
        self.image_processor = transforms.ToTensor()

    
    def proc_prompt(self, prompt):
        return '<img>0</img>' + 'Question: {} Answer:'.format(prompt)
    
    @torch.no_grad()
    def raw_generate(self, image, prompt, temperature=0.2, max_new_tokens=512):
        # image = get_image(image)
        image = image[0]
        prompt = self.proc_prompt(prompt)
        input_ids = self.tokenizer(prompt, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids
        images = [image] # self.image_processor(image).unsqueeze(0)

        with torch.inference_mode() and torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                images=images, # .cuda(),
                do_sample=False,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eod_id,
                eos_token_id=self.tokenizer.eod_id,
        )
        answer = self.tokenizer.decode(output_ids[0][input_ids.size(1):].cpu(), skip_special_tokens=True)
        outputs = answer.strip()

        return outputs
    
    @torch.no_grad()
    def raw_batch_generate(self, image_list, question_list, temperature=0.2, max_new_tokens=30):
        outputs = [self.raw_generate(img, question, temperature, max_new_tokens) for img, question in zip(image_list, question_list)]

        return outputs
    
    @torch.no_grad()
    def raw_predict(self, image, prompt, candidates, likelihood_reduction='sum'):
        image = image[0] # get_image(image)
        prompt = self.proc_prompt(prompt)
        input_ids = self.tokenizer(prompt, return_tensors='pt', padding='longest')
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids
        images = [image]
        loss_fct = nn.CrossEntropyLoss()

        with torch.inference_mode() and torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                images=images,
                use_cache=True,
            )
            question_logits = outputs.logits
            question_past_key_values = outputs.past_key_values
            loss_list = []
            for opt in candidates:
                opt_ids = self.tokenizer(' '+opt, return_tensors='pt').input_ids.to(self.device)
                output_opt = self.model(
                    input_ids=opt_ids,
                    use_cache=True,
                    attention_mask=torch.ones(1, question_logits.shape[1]+opt_ids.shape[1], device=self.device),
                    past_key_values=question_past_key_values
                )
                logits = torch.cat([question_logits[:, -1:], output_opt.logits[:, :-1]], 1)
                logits = logits.view(-1, self.model.config.vocab_size)
                labels = opt_ids.view(-1)
                loss = loss_fct(logits, labels)
                loss_list.append(loss)
        option_chosen = torch.stack(loss_list).argmin().cpu().item()


        if likelihood_reduction == 'none':
            raise NotImplementedError
            return input_ids, neg_likelihood
        # output_class_ranks = torch.argsort(neg_likelihood, dim=-1)[0].item()

        return option_chosen
    
    @torch.no_grad()
    def raw_batch_predict(self, image_list, question_list, candidates, likelihood_reduction='sum'):
        preds = [self.raw_predict(image, question, cands, likelihood_reduction=likelihood_reduction) for image, question, cands in zip(image_list, question_list, candidates)]

        return preds
    
    def forward(self, image, prompt, candidates=None, temperature=0.2, max_new_tokens=30, likelihood_reduction='sum'):
        if self.inference_method == 'generation':
            return self.raw_batch_generate(image, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        elif self.inference_method == 'likelihood':
            assert candidates is not None, "the candidate list should be set for likelihood inferecne!"
            return self.raw_batch_predict(image, prompt, candidates, likelihood_reduction=likelihood_reduction)
        else:
            raise NotImplementedError
    
def get_monkey(model_config=None):
    model_args = {}
    if model_config is not None:
        valid_args = ['model_path', 'device', 'half', 'inference_method']
        target_args = ['model_path', 'device', 'half', 'inference_method']
        for i,arg in enumerate(valid_args):
            if arg in model_config:
                model_args[target_args[i]] = model_config[arg]
    model = MonkeyInterface(**model_args)
    # first_query_process_fn = model.get_first_query_process()
    
    # proc = ConvSingleChoiceProcessor(' ', sep2='\n', roles=['Question', 'Answer'], system_msg='<img>0</img>', \
    #                                  first_query_fn=None, init_conv=None, \
    #                                  sep_style='two', infer_method=model_args['inference_method'],
    #                                  response_prefix=None)
    # proc = SingleChoiceProcessor(' ', '\n', roles=['Question', 'Answer'], infer_method=model_args['inference_method'])
    return model, False, False

if __name__=='__main__':
    model = MonkeyInterface()