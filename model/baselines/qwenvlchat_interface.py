from PIL import Image
import requests
from PIL import Image
from io import BytesIO
# from transformers import TextStreamer
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.generation import GenerationConfig
import torch
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from typing import List, Tuple

def make_context(
    tokenizer: PreTrainedTokenizer,
    query: str,
    history: List[Tuple[str, str]] = None,
    system: str = "",
    max_window_size: int = 6144,
    chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set(tokenizer.IMAGE_ST)
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str(
                    "assistant", turn_response
                )
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = (
                    f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
                )
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"

            current_context_size = (
                len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
            nl_tokens
            + im_start_tokens
            + _tokenize_str("user", query)[1]
            + im_end_tokens
            + nl_tokens
            + im_start_tokens
            + tokenizer.encode("assistant")
            + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens

class Qwenvlchat_Interface(nn.Module):
    def __init__(self, model_path="facebook/opt-350m", device=None, half=False, inference_method='generation') -> None:
        super(Qwenvlchat_Interface, self).__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)  

        # self.pretrained_ckpt = model_path
        # self.prec_half = half

        torch.manual_seed(1234)

        # Note: The default behavior now has injection attack prevention off.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # use bf16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
        # use fp16
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
        # use cpu only
        # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
        # use cuda device
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, fp16=True, device_map="cpu").eval()

        self.model.to(self.device).half()

        # Specify hyperparameters for generation
        self.model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

        # setup the inference method
        self.inference_method = inference_method

    # def get_conv(self):
    #     return self.conv
    
    # def get_first_query_process(self):
    #     if getattr(self.model.config, 'mm_use_im_start_end', False):
    #         return lambda qs: DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    #     else:
    #         return lambda qs: DEFAULT_IMAGE_TOKEN + '\n' + qs
    def get_stop_words_ids(self, chat_format, tokenizer):
        if chat_format == "raw":
            stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
        elif chat_format == "chatml":
            stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
        else:
            raise NotImplementedError(f"Unknown chat format {chat_format!r}")
        return stop_words_ids
    
    @torch.no_grad()
    def raw_generate(self, image, prompt, temperature=0.2, max_new_tokens=30):
        # image_text = f'Picture 1: <img>{image}</img>\n'
        # #query = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<img>{image}</img>{prompt}<|im_end|>\n<|im_start|>assistant\n'
        # # query = tokenizer()
        # # modified_query = f'Picture 1: <img>{image}</img>\n{prompt}'
        # parts = prompt.split("user\n", 1)

        # modified_query = parts[0] + f"user\n{image_text}" + parts[1]
        # # print(modified_query)

        # inputs = self.tokenizer(modified_query, return_tensors='pt')
        # # print(inputs.input_ids.shape)
        # inputs = inputs.to(self.model.device)
        # stop_words_ids = []
        # stop_words_ids.extend(self.get_stop_words_ids(
        #     "chatml", self.tokenizer
        # ))
        query = self.tokenizer.from_list_format([{'image': image[0]}, {'text': prompt}])
        response, history = self.model.chat(self.tokenizer, query=query, history=None)
        # a = response.split(':')[-1]
        return response
    
    @torch.no_grad()
    def raw_batch_generate(self, image_list, question_list, temperature=0.2, max_new_tokens=30):
        outputs = [self.raw_generate(img, question, temperature, max_new_tokens) for img, question in zip(image_list, question_list)]

        return outputs
    
    @torch.no_grad()
    def raw_predict(self, image, prompt, candidates, likelihood_reduction='mean'):
        query = self.tokenizer.from_list_format([{'image': image[0]}, {'text': prompt}])
        raw_text, context_tokens = make_context(
            self.tokenizer, query, history=[], system='You are a helpful assistant.',max_window_size=self.model.generation_config.max_window_size,
            chat_format=self.model.generation_config.chat_format,
        )
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        self.tokenizer.eos_token_id = self.tokenizer.eod_id
        input_ids = torch.tensor([context_tokens]).to(self.model.device)
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
        candidates_ids = candidates_tokens.input_ids[:, :] # remove the <s> token
        candidates_att = candidates_tokens.attention_mask[:, :] # remove the <s> token
        # mask the LM targets with <pad>
        cand_targets = candidates_ids.clone()
        cand_targets = cand_targets.masked_fill(cand_targets == self.tokenizer.pad_token_id, -100)
        # mask the targets for inputs part
        targets = torch.cat([-100*torch.ones_like(input_ids), cand_targets], dim=1)
        # concatenate the inputs for the model
        input_ids = torch.cat([input_ids, candidates_ids], dim=1)
        attention_mask = torch.cat([attention_mask, candidates_att], dim=1)

        # with torch.inference_mode():
        #     outputs = self.model(
        #         input_ids,
        #         # images=image.repeat_interleave(len(candidates), dim=0),
        #         attention_mask=attention_mask,
        #         labels=targets,
        #         return_dict=False,
        #         # likelihood_reduction=likelihood_reduction
        #     )
        # # print(outputs)

        # loss = outputs.loss.reshape(input_ids.shape[0], -1)
        # if likelihood_reduction == 'sum':
        #     loss = loss.sum(1)
        # elif likelihood_reduction == 'mean':
        #     valid_num_targets = (loss > 0).sum(1)
        #     loss = loss.sum(1) / valid_num_targets
        # elif likelihood_reduction == 'none':
        #     loss = loss
        # else:
        #     raise ValueError
        # # return loss
        # neg_likelihood = loss
        # if likelihood_reduction == 'none':
        #     return input_ids, neg_likelihood
        # output_class_ranks = torch.argsort(neg_likelihood, dim=-1)[0].item()

        # return output_class_ranks
        from torch.nn import CrossEntropyLoss
        with torch.inference_mode():
            transformer_outputs = self.model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask)
            
        hidden_states = transformer_outputs[0]

        lm_logits = self.model.lm_head(hidden_states)

        loss = None
        labels = targets
        if labels is not None:
            labels = labels.to(lm_logits.device)
            # shift_logits = lm_logits[..., :-1, :].contiguous()
            output = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # loss_fct = CrossEntropyLoss()
            # loss = loss_fct(
            #     shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            # )                from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss(reduction='none')
            vocab_size = output.shape[-1]
            shift_logits = output.view(-1, vocab_size)
            shift_labels_ids = shift_labels.view(-1)
                
            shift_labels_ids = shift_labels_ids.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels_ids)
            loss = loss.view(output.size(0), -1)
            
            if likelihood_reduction == 'sum':
                loss = loss.sum(1)
                # print(loss)
            elif likelihood_reduction == 'mean':
                valid_num_targets = (loss > 0).sum(1)
                loss = loss.sum(1) / valid_num_targets
            elif likelihood_reduction == 'none':
                loss = loss
                return loss
            else:
                raise ValueError
            output_class_ranks = torch.argsort(loss, dim=-1)[0].item()

            return output_class_ranks

    
    @torch.no_grad()
    def raw_batch_predict(self, image_list, question_list, candidates, likelihood_reduction='mean'):
        preds = [self.raw_predict(image, question, cands, likelihood_reduction=likelihood_reduction) for image, question, cands in zip(image_list, question_list, candidates)]

        return preds
    
    def forward(self, image, prompt, candidates=None, temperature=0.2, max_new_tokens=30, likelihood_reduction='mean'):
        if self.inference_method == 'generation':
            return self.raw_batch_generate(image, prompt, temperature=temperature, max_new_tokens=max_new_tokens)
        elif self.inference_method == 'likelihood':
            assert candidates is not None, "the candidate list should be set for likelihood inferecne!"
            return self.raw_batch_predict(image, prompt, candidates, likelihood_reduction=likelihood_reduction)
        else:
            raise NotImplementedError
    
def get_qwenvlchat(model_config=None):
    model_args = {}
    if model_config is not None:
        valid_args = ['model_path', 'inference_method']
        target_args = ['model_path', 'inference_method']
        for i,arg in enumerate(valid_args):
            if arg in model_config:
                model_args[target_args[i]] = model_config[arg]
    model = Qwenvlchat_Interface(**model_args)
    # conv = model.get_conv()
    # first_query_process_fn = model.get_first_query_process()
    # if conv.sep_style.name == 'SINGLE':
    #     sep_style = 'one'
    # elif conv.sep_style.name == 'TWO':
    #     sep_style = 'two'
    # elif conv.sep_style.name == 'LLAMA_2':
    #     sep_style = 'llama_2'
    # else:
    #     raise NotImplementedError
    # query = f'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<img>{img_item_path}</img>{text}<|im_end|>\n<|im_start|>assistant\n'

    # proc = ConvSingleChoiceProcessor(conv.sep, sep2=conv.sep2, roles=conv.roles, system_msg=conv.system, \
    #                                  first_query_fn=first_query_process_fn, init_conv=conv.messages, \
    #                                  sep_style=sep_style, infer_method=model_args['inference_method'],
    #                                  response_prefix='The answer is')
    return model, False, True

if __name__=='__main__':
    model = Qwenvlchat_Interface()