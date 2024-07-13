
from PIL import Image
import requests
from PIL import Image
from io import BytesIO
# from transformers import TextStreamer
# from .utils import get_image
import torch
import torch.nn as nn

from mplug_owl2.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from mplug_owl2.conversation import conv_templates, SeparatorStyle
from mplug_owl2.model.builder import load_pretrained_model
from mplug_owl2.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

class Mplugowl2_Interface(nn.Module):
    def __init__(self, model_base=None, model_path="facebook/opt-350m", device=None, half='fp16', inference_method='generation') -> None:
        super(Mplugowl2_Interface, self).__init__()
        model_name = get_model_name_from_path(model_path)
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path, model_base, \
                                                                                model_name, device_map=device)
        
        # add pad to tokenizer if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # convert to the device
        if half == 'bf16':
            self.model.to(torch.bfloat16)
            self.dtype = torch.bfloat16
        elif half == 'fp16':
            self.model.to(torch.float16)
            self.dtype = torch.float16
        self.model.to(self.device)
        
        self.conv = conv_templates["mplug_owl2"].copy()

        self.roles = self.conv.roles

        # setup the inference method
        self.inference_method = inference_method
        self.first_query_process_fn = lambda qs: DEFAULT_IMAGE_TOKEN + '\n' + qs

    def get_conv(self):
        return self.conv
    
    # def get_first_query_process(self):
    #     if getattr(self.model.config, 'mm_use_im_start_end', False):
    #         return lambda qs: DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    #     else:
    #         return lambda qs: DEFAULT_IMAGE_TOKEN + '\n' + qs
    
    @torch.no_grad()
    def raw_generate(self, image, prompt, temperature=0.2, max_new_tokens=30):
        # image = get_image(image)
        max_edge = max(image[0].size) # We recommand you to resize to squared image for BEST performance.
        image = image[0].resize((max_edge, max_edge))
        query = self.first_query_process_fn(prompt)
        conv = self.conv.copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(self.device)
        # if getattr(self.model.config, 'mm_use_im_start_end', False):
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        # inp = DEFAULT_IMAGE_TOKEN + prompt
        # self.conv.append_message(conv.roles[0], inp)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

        # setup the stopping criteria
        stop_str = self.conv.sep if self.conv.sep_style != SeparatorStyle.TWO else self.conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        return outputs
    
    @torch.no_grad()
    def raw_batch_generate(self, image_list, question_list, temperature=0.2, max_new_tokens=30):
        outputs = [self.raw_generate(img, question, temperature, max_new_tokens) for img, question in zip(image_list, question_list)]

        return outputs
    
    @torch.no_grad()
    def raw_predict(self, image, prompt, candidates, likelihood_reduction='sum'):
        query = self.first_query_process_fn(prompt)
        conv = self.conv.copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        max_edge = max(image[0].size) # We recommand you to resize to squared image for BEST performance.
        image = image[0].resize((max_edge, max_edge))
        image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().to(self.device)
        # if getattr(self.model.config, 'mm_use_im_start_end', False):
        #     qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs/root/LLM-V-Bench/models/build_scripts
        # else:
        #     qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)

        # prepare inputs for the input part
        input_ids = input_ids.repeat_interleave(len(candidates), dim=0)
        attention_mask = torch.ones_like(input_ids, dtype=input_ids.dtype, device=input_ids.device)
        
        # tokenize the candidates
        current_padding_side = self.tokenizer.padding_side
        current_truncation_side = self.tokenizer.truncation_side
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'
        candidates_tokens = self.tokenizer(
            [cand for cand in candidates],
            return_tensors='pt',
            padding='longest'
        ).to(self.device)
        self.tokenizer.padding_side = current_padding_side
        self.tokenizer.truncation_side = current_truncation_side

        # construct the inputs_ids and LM targets
        candidates_ids = candidates_tokens.input_ids[:, 1:] # remove the <s> token
        candidates_att = candidates_tokens.attention_mask[:, 1:] # remove the <s> token
        # mask the LM targets with <pad>
        cand_targets = candidates_ids.clone()
        cand_targets = cand_targets.masked_fill(cand_targets == self.tokenizer.pad_token_id, -100)
        # mask the targets for inputs part
        targets = torch.cat([-100*torch.ones_like(input_ids), cand_targets], dim=1)
        # concatenate the inputs for the model
        input_ids = torch.cat([input_ids, candidates_ids], dim=1)
        attention_mask = torch.cat([attention_mask, candidates_att], dim=1)

        with torch.inference_mode():
            outputs = self.model(
                input_ids,
                images=image.repeat_interleave(len(candidates), dim=0),
                attention_mask=attention_mask,
                labels=targets,
                return_dict=True
            )
        logits = outputs.logits
        labels = targets
        option_lens = candidates_ids.shape[1]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., -(1+option_lens):-1, :].contiguous()
            shift_labels = labels[..., -option_lens:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss = loss.view(logits.shape[0], -1)
        if likelihood_reduction == 'sum':
            loss = loss.sum(1)
        elif likelihood_reduction == 'mean':
            loss = loss.mean(1)
        else:
            loss = loss
        neg_likelihood = loss
        if likelihood_reduction == 'none':
            return input_ids, neg_likelihood
        output_class_ranks = torch.argsort(neg_likelihood, dim=-1)[0].item()

        return output_class_ranks
    
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
    
def get_mplugowl2(model_config=None):
    model_args = {}
    if model_config is not None:
        valid_args = ['model_path', 'model_type', 'device', 'half', 'inference_method']
        target_args = ['model_path', 'model_base', 'device', 'half', 'inference_method']
        for i,arg in enumerate(valid_args):
            if arg in model_config:
                model_args[target_args[i]] = model_config[arg]
    model = Mplugowl2_Interface(**model_args)
    conv = model.get_conv()
    first_query_process_fn = lambda qs: DEFAULT_IMAGE_TOKEN + '\n' + qs
    return model, False, False

if __name__=='__main__':
    model = Mplugowl2_Interface()