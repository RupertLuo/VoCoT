from transformers import AutoModelForCausalLM
from dataclasses import dataclass
from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from typing import Dict, List
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
# from transformers import TextStreamer
# from .utils import get_image
import torch
import torch.nn as nn
# from utils.preprocessors import BaseProcessor, SingleChoiceProcessor, ConvSingleChoiceProcessor

class DictOutput(object):
    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


@dataclass
class VLChatProcessorOutput(DictOutput):
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)
    
@dataclass
class BatchedVLChatProcessorOutput(DictOutput):
    sft_format: List[str]
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    attention_mask: torch.Tensor
    images_seq_mask: torch.BoolTensor
    images_emb_mask: torch.BoolTensor

    def to(self, device, dtype=torch.bfloat16):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.images_seq_mask = self.images_seq_mask.to(device)
        self.images_emb_mask = self.images_emb_mask.to(device)
        self.pixel_values = self.pixel_values.to(device=device, dtype=dtype)
        return self

class DeepSeekVLInterface(nn.Module):
    def __init__(self, model_base=None, model_path="deepseek-ai/deepseek-vl-7b-chat", device=None, half=False, inference_method='generation') -> None:
        super(DeepSeekVLInterface, self).__init__()
        self.processor = VLChatProcessor.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.tokenizer = self.processor.tokenizer
        if half=='fp16':
            dtype = torch.float16 # if torch.cuda.is_bf16_supported() else torch.float16
        elif half=='bf16':
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        self.dtype = dtype
        self.model = self.model.to(dtype).to(device)
        self.image_processor = self.processor.image_processor
        
        # convert to the device
        self.model.to(self.device)
        self.force_batchify = True
        
        self.system_prompt  = (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
        )
        self.stop_token_ids = [100001],
        self.stop_str = ["User:", "<｜end▁of▁sentence｜>"]

        # setup the inference method
        self.inference_method = inference_method
        self.query_process = self.get_first_query_process()

    def get_conv(self):
        return self.conv
    
    def get_first_query_process(self):
        return lambda qs: '<image_placeholder>' + qs
    
    def process_item(self, text, image):
        input_ids = self.tokenizer.encode(text)
        input_ids = torch.LongTensor(input_ids)

        # add image tokens to the input_ids
        image_token_mask: torch.BoolTensor = input_ids == self.processor.image_id
        image_indices = image_token_mask.nonzero()
        input_ids, num_image_tokens = self.processor.add_image_token(
            image_indices=image_indices,
            input_ids=input_ids,
        )

        # load images
        images_outputs = self.image_processor([image], return_tensors="pt")
        prepare =  VLChatProcessorOutput(
            sft_format=text,
            input_ids=input_ids,
            pixel_values=images_outputs.pixel_values.to(self.dtype),
            num_image_tokens=num_image_tokens,
        )
        if self.force_batchify:
            prepare = self.processor.batchify([prepare])
        return prepare.to(self.device, self.dtype)
    
    
    @torch.no_grad()
    def raw_generate(self, image, prompt, temperature=0.2, max_new_tokens=512):
        # image = get_image(image)
        image = image[0]
        prompt = self.query_process(prompt)
        prepare_inputs = self.process_item(prompt, image)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)

        with torch.inference_mode() and torch.no_grad():
            output_ids = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True
            )
        answer = self.tokenizer.decode(output_ids[0].cpu().tolist(), skip_special_tokens=True)
        outputs = answer.strip()

        return outputs
    
    @torch.no_grad()
    def raw_batch_generate(self, image_list, question_list, temperature=0.2, max_new_tokens=512):
        outputs = [self.raw_generate(img, question, temperature, max_new_tokens) for img, question in zip(image_list, question_list)]

        return outputs
    
    @torch.no_grad()
    def raw_predict(self, image, prompt, candidates, likelihood_reduction='sum'):
        image = image[0]
        prompt = self.query_process(prompt)
        prepare_inputs = self.process_item(prompt, image)
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        if likelihood_reduction == 'none' or likelihood_reduction =='sum':
            loss_fct = nn.CrossEntropyLoss(reduction='none')
        else:
            loss_fct = nn.CrossEntropyLoss()

        with torch.inference_mode() and torch.no_grad():
            outputs = self.model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                use_cache=True,
            )
            question_logits = outputs.logits
            question_past_key_values = outputs.past_key_values
            loss_list = []
            for opt in candidates:
                # " "+ opt here since deepseek will not append space between bos and the first token
                opt_ids = torch.LongTensor(self.tokenizer.encode(' '+ opt)[1:]).unsqueeze(0).to(self.device)
                output_opt = self.model.language_model(
                    inputs_embeds=self.model.language_model.get_input_embeddings()(opt_ids),
                    use_cache=True,
                    attention_mask=torch.ones(1, question_logits.shape[1]+opt_ids.shape[1], device=self.device),
                    past_key_values=question_past_key_values
                )
                logits = torch.cat([question_logits[:, -1:], output_opt.logits[:, :-1]], 1)
                logits = logits.view(-1, self.model.language_model.config.vocab_size)
                labels = opt_ids.view(-1)
                loss = loss_fct(logits, labels)
                if likelihood_reduction == 'sum':
                    loss = loss.sum()
                loss_list.append(loss)
        if likelihood_reduction == 'none':
            return loss_list
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
    
    def forward(self, image, prompt, candidates=None, temperature=0.2, max_new_tokens=512, likelihood_reduction='sum'):
        if self.inference_method == 'generation':
            return self.raw_batch_generate(image, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        elif self.inference_method == 'likelihood':
            assert candidates is not None, "the candidate list should be set for likelihood inferecne!"
            return self.raw_batch_predict(image, prompt, candidates, likelihood_reduction=likelihood_reduction)
        else:
            raise NotImplementedError
    
def get_deepseek(model_config=None):
    model_args = {}
    if model_config is not None:
        valid_args = ['model_path', 'device', 'half', 'inference_method']
        target_args = ['model_path', 'device', 'half', 'inference_method']
        for i,arg in enumerate(valid_args):
            if arg in model_config:
                model_args[target_args[i]] = model_config[arg]
    model = DeepSeekVLInterface(**model_args)
    first_query_process_fn = model.get_first_query_process()
    return model, False, False

if __name__=='__main__':
    model = DeepSeekVLInterface()