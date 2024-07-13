from typing import List, Optional, Tuple, Union
from xml.dom.expatbuilder import parseString
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                        LlamaConfig, LlamaModel, LlamaForCausalLM, CLIPImageProcessor
# from .modeling_llama import LlamaConfig, LlamaModel, LlamaForCausalLM
# from .modeling_mistral import MistralForCausalLM, MistralConfig, MistralModel
from transformers import MistralForCausalLM, MistralConfig, MistralModel

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput

from ..vision_encoder.builder import build_vision_encoder
from ..vision_generator.builder import build_vision_generator
from ..front_projector.builder import build_front_projector
from ..behind_projector.builder import build_behind_projector
from .volcano_base import VolCanoMetaForCausalLM
from transformers import GenerationMixin

from PIL import Image
from constants import *
import random
from utils.util import rank_0_print
import torch.nn.functional as F
from diffusers.models.vae import DiagonalGaussianDistribution
from transformers import StoppingCriteria, StoppingCriteriaList
from utils.eval_util import extract_box_str
from locals.datasets.utils.box_utils import *

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

def resize_image_to_square(image):
    width, height = image.size
    max_side = max(width, height)
    image = image.resize((max_side, max_side))
    return image

def split_tensor_func(tensor, dim=0):
    num_splits = tensor.shape[dim]
    return [f.squeeze(dim) for f in tensor.split([1]*num_splits, dim=dim)]

@dataclass
class VolCanoCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    regression_loss: Optional[torch.FloatTensor] = None
    text_loss: Optional[torch.FloatTensor] = None

IGNORE_INDEX = -100
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        return super().forward(x)


class VolCanoMistralConfig(MistralConfig):
    model_type = "VolCanoMistral"


class VolCanoMistralModel(MistralModel):
    config_class = VolCanoMistralConfig

    def __init__(self, config: MistralConfig):
        super(VolCanoMistralModel, self).__init__(config)
        

class VolCanoMistralForCausalLM(MistralForCausalLM, VolCanoMetaForCausalLM):
    config_class = VolCanoMistralConfig

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)
        
        self.model = VolCanoMistralModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.init_vision_model()

        if hasattr(self, 'vision_generator'):

            zero_img_feature = torch.zeros((1, IMG_TOKEN_NUM, config.hidden_size))
            self.register_buffer('zero_img_feature', zero_img_feature, persistent=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.output_img_id = -100
        self.regression_weight = 1.0
        self.no_bind = False
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        captions=[None],
        output_image_feature=None,
        output_images = None,
        output_cond_images = None,
        output_cond_img_mask = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        input_images: Optional[torch.FloatTensor] = None,
        image_label_masks: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        item_id: Optional[bool] = None,
        box: Optional[torch.Tensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # return torch.zeros(1).to(self.device).to(self.dtype)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels, visual_labels, visual_label_masks = self.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids ,attention_mask, past_key_values, labels = labels, images = input_images, image_label_masks=image_label_masks, inputs_embeds=inputs_embeds, box=box)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(self.dtype)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids = position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        
        hidden_states = outputs[0]
        logits = getattr(self.lm_head,'modules_to_save.default',self.lm_head)(hidden_states)
        loss = None
        text_loss = None
        # compute text loss
        if labels is not None:
            # Shift so that tokens < n predict n

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            text_loss = loss_fct(shift_logits, shift_labels)
            # rank_0_print(text_loss)
            # rank_0_print(labels)
            
        # regression loss
        if visual_labels is not None and text_loss is not None:
            if visual_label_masks.sum() == 0:
                # # no target images
                # if text_loss is not None:
                #     regression_loss = torch.zeros_like(text_loss)
                # else:
                #     regression_loss = None
                last_hidden_state = hidden_states
                target_img_hidden_states = torch.masked_select(last_hidden_state[..., :-1, :], (visual_label_masks[:, 1:]>0).unsqueeze(-1)).reshape(-1, hidden_states.shape[-1])
                predict_image_feat = self.behind_projector(target_img_hidden_states)
                target_visual_labels = torch.masked_select(visual_labels, (visual_label_masks>0).unsqueeze(-1)).reshape(-1, visual_labels.shape[-1])
                regression_loss = F.mse_loss(predict_image_feat, target_visual_labels, reduction='none').sum()
                loss = self.regression_weight*regression_loss + (2 - self.regression_weight) * text_loss
            else:
                last_hidden_state = hidden_states
                target_img_hidden_states = torch.masked_select(last_hidden_state[..., :-1, :], (visual_label_masks[:, 1:]>0).unsqueeze(-1)).reshape(-1, hidden_states.shape[-1])
                predict_image_feat = self.behind_projector(target_img_hidden_states)
                target_visual_labels = torch.masked_select(visual_labels, (visual_label_masks>0).unsqueeze(-1)).reshape(-1, visual_labels.shape[-1])
                regression_loss = F.mse_loss(predict_image_feat, target_visual_labels)

                if self.diffusion_loss:
                    num_output_images = output_images.shape[0]
                    predict_image_feat = predict_image_feat.reshape(-1, self.n_query, predict_image_feat.shape[-1])
                    assert num_output_images == (predict_image_feat.shape[0]), 'the output images must match the images in sequences'
                    random_probs = torch.rand(num_output_images).to(output_images.device).unsqueeze(-1).unsqueeze(-1)
                    zero_image_feature = self.encode_img(torch.zeros(1, 3, self.vision_encoder.image_size, self.vision_encoder.image_size).to(device=output_images.device, dtype=output_images.dtype))[2]
                    if USE_CFG:
                        diffusion_input_feature = torch.where(random_probs < 0.1, zero_image_feature, predict_image_feat)
                    else:
                        diffusion_input_feature = predict_image_feat

                    cond_img_mask = None
                    output_cond_images = None

                    if output_image_feature is None:
                        image_loss = self.compute_image_loss(diffusion_input_feature, output_images, output_cond_image=output_cond_images, cond_img_mask=cond_img_mask)
                    else:
                        image_loss = self.compute_image_loss(diffusion_input_feature, None, output_cond_image=output_cond_images, cond_img_mask=cond_img_mask, output_image_feature=output_image_feature)

                    loss = regression_loss + text_loss + image_loss
                else:
                    # print("regression loss:{:.5f}".format(regression_loss.item()) + ' text loss:{:.5f}'.format(text_loss.item()))
                    loss = self.regression_weight*regression_loss + (2 - self.regression_weight) * text_loss
        else:
            loss = text_loss
            if text_loss is not None:
                regression_loss = torch.zeros_like(text_loss)
            else:
                regression_loss = None


        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return VolCanoCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            regression_loss=regression_loss,
            text_loss=text_loss
        )

    def compute_snr(self,timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = self.noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    def compute_image_loss(self, mapping_feature, output_image, output_cond_image=None, cond_img_mask=None, output_image_feature=None):
        if output_image_feature is not None:
            latents = DiagonalGaussianDistribution(output_image_feature).sample()
        else:
            if len(output_image.shape) == 3:
                output_image = output_image.unsqueeze(0)

            latents = self.vision_generator.vae.encode(output_image).latent_dist.sample()
        if self.image_condition:
            assert output_cond_image is not None, "the current model requires image as conditions"
            # mask the uncond (can be accelerated here TODO!)
            image_cond_latents = self.vision_generator.vae.encode(output_cond_image).latent_dist.mode()
            cond_img_mask = cond_img_mask.to(image_cond_latents.dtype).unsqueeze(1).unsqueeze(2).unsqueeze(3)
            image_cond_latents = cond_img_mask*image_cond_latents
        
        latents = latents * self.vision_generator.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        target = noise

        if self.image_condition:
            # concatenate the image condition in the channels
            noisy_latents = torch.cat([noisy_latents, image_cond_latents], dim=1)
        unet_added_conditions = {}

        if self.sd_add_args:
            # time_ids = torch.LongTensor(original_size + crop_info + [height, width]).to(mapping_feature.device)
            time_ids = torch.LongTensor([1024, 1024, 0, 0, 1024, 1024]).to(mapping_feature.device)
            unet_added_conditions["time_ids"] = time_ids.repeat([bsz])
            unet_added_conditions["text_embeds"] = torch.mean(mapping_feature, dim=1)
        model_pred = self.vision_generator.unet(noisy_latents, 
                                                timesteps, mapping_feature,
                                                added_cond_kwargs=unet_added_conditions).sample


        if self.config.snr_loss:
            snr = self.compute_snr(timesteps)
            mse_loss_weights = (
                torch.stack([snr, 5 * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            # We first calculate the original loss. Then we mean over the non-batch dimensions and
            # rebalance the sample-wise losses with their respective loss weights.
            # Finally, we take the mean of the rebalanced loss.
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        else:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        return loss
    
    def encode_caption(self, caption, length, inference=False):
        # add_special_tokens = False
        # if len(caption) == 0:
        add_special_tokens = True
        text_inputs = self.vision_generator.sd_tokenizer(
                caption,
                padding="max_length",
                max_length=length,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=add_special_tokens
            ).to(self.device)
        # text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        prompt_embeds = self.vision_generator.sd_text_encoder(**text_inputs)[0]
        return prompt_embeds

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        original_input_ids = input_ids.clone()
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "input_images": kwargs.get("input_images", None),
                "box": kwargs.get("box", None),
                "image_label_masks": kwargs.get("image_label_masks", None),
            }
        )

        if 'input_ids' in model_inputs:
            new_token_ids = model_inputs['input_ids'][:, -1:]
            if new_token_ids == self.boi_token:
                #Generated the image token, force add all the image tokens
                next_inputs_embeds, current_target_image_embeds = self.generate_image(model_inputs)
                self.to_generate_images.append(current_target_image_embeds)
                model_inputs['input_ids'] = None
                model_inputs['inputs_embeds'] = next_inputs_embeds
                all_img_tokens_mask = torch.ones(1, self.n_query).to(device=model_inputs['attention_mask'].device, dtype=model_inputs['attention_mask'].dtype)
                model_inputs['attention_mask'] = torch.cat([model_inputs['attention_mask'], all_img_tokens_mask], dim=1)
            if new_token_ids == self.eoc_token_id and not self.no_bind:
                # need bounding box detection and use box align
                if self.sub_image_bind:
                    next_inputs_embeds, query_len = self.generate_sub_image(model_inputs, original_input_ids)
                else:
                    next_inputs_embeds, query_len = self.generate_box(model_inputs, original_input_ids)
                model_inputs['input_ids'] = None
                model_inputs['inputs_embeds'] = next_inputs_embeds
                all_img_tokens_mask = torch.ones(1, query_len).to(device=model_inputs['attention_mask'].device, dtype=model_inputs['attention_mask'].dtype)
                model_inputs['attention_mask'] = torch.cat([model_inputs['attention_mask'], all_img_tokens_mask], dim=1)
        return model_inputs

    def generate_box(self, model_inputs, original_input_ids):
        assert original_input_ids.shape[0] == 1
        if self.cache_images is None:
            self.cache_images = self.encode_img(model_inputs['input_images'][0][:1])[0]
        valid_start_ind = torch.where(original_input_ids==self.boc_token_id)[1].tolist()[-1]
        current_box_text = self.tokenizer.decode(original_input_ids[0, valid_start_ind:])
        current_box = torch.tensor(extract_box_str(current_box_text, mistral=True), dtype=self.dtype, device=self.cache_images.device)
        if current_box is None:
            print('fail to detect correct box from {}'.format(current_box_text))
            raise ValueError
        box_feat = self.box_align(self.cache_images[0], current_box.unsqueeze(0))[0]
        init_inputs_embeds = self.get_input_embeddings()(model_inputs['input_ids'])
        next_inputs_embeds = torch.cat([init_inputs_embeds, box_feat.unsqueeze(0)], dim=1)
        return next_inputs_embeds, box_feat.shape[0]

    def generate_sub_image(self, model_inputs, original_input_ids):
        assert original_input_ids.shape[0] == 1
        assert self.cache_raw_image is not None
        valid_start_ind = torch.where(original_input_ids==self.boc_token_id)[1].tolist()[-1]
        current_box_text = self.tokenizer.decode(original_input_ids[0, valid_start_ind:])
        current_box = extract_box_str(current_box_text, mistral=True)
        if current_box is None:
            print('fail to detect correct box from {}'.format(current_box_text))
            raise ValueError
        # box_feat = self.box_align(self.cache_images[0], current_box.unsqueeze(0))[0]
        w, h = self.cache_raw_image.size
        x_min, y_min, x_max, y_max = current_box
        x_min = x_min*w
        y_min = y_min*h
        x_max = x_max*w
        y_max = y_max*h
        sub_image = self.cache_raw_image.crop((x_min, y_min, x_max, y_max))
        sub_image = self.image_processor(resize_image_to_square(sub_image)).unsqueeze(0).to(dtype=self.dtype, device=original_input_ids.device)
        box_feat = self.encode_img(sub_image)[0][0]
        init_inputs_embeds = self.get_input_embeddings()(model_inputs['input_ids'])
        next_inputs_embeds = torch.cat([init_inputs_embeds, box_feat.unsqueeze(0)], dim=1)
        return next_inputs_embeds, box_feat.shape[0]
    
    def generate_image(self, model_inputs):
        input_ids = model_inputs['input_ids']
        past_key_values = model_inputs['past_key_values']
        use_cache = model_inputs['use_cache']
        attention_mask = model_inputs['attention_mask']
        bs = input_ids.shape[0]
        target_image_embeds = None
        init_inputs_embeds = self.get_input_embeddings()(input_ids)
        for num_img_token in range(self.n_query):
            if num_img_token == 0:
                inputs_embeds = init_inputs_embeds
            else:
                inputs_embeds = torch.cat([init_inputs_embeds, self.front_mm_projector(target_image_embeds)], dim=1)
            if past_key_values is None:
                target_shape = num_img_token + 1
            else:
                target_shape = past_key_values[-1][-1].shape[-2] + num_img_token + 1
            attention_mask = torch.cat((attention_mask, torch.ones(
                (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )), dim=1)
            sentence_length = torch.sum(attention_mask, dim=1).item()
            position_ids = torch.arange(sentence_length-(num_img_token+1), sentence_length).unsqueeze(0).to(attention_mask.device)
            outputs = self.model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states[-1]
            target_image_embeds = hidden_states[:, -(num_img_token+1):, :]
            target_image_embeds = self.behind_projector(target_image_embeds)
        inputs_embeds = torch.cat([init_inputs_embeds, self.front_mm_projector(target_image_embeds)], dim=1)
        return inputs_embeds, target_image_embeds
    
    def flatten_hidden(self, hidden_state_list):
        last_hidden_state = []
        for hid_st in hidden_state_list:
            last_hidden_state.append(hid_st[-1])
        last_hidden_state = torch.cat(last_hidden_state,dim=1)
        return last_hidden_state

    def condition_completion(self, input_dict, temperature=0.2, max_new_tokens=128, guidance_scale=7.5, avoid_image_gen=False, **kwargs):
        
        self.to_generate_images = []
        self.cache_images = None
        self.boi_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_BOI_TOKEN])[0]
        if self.sub_image_bind:
            self.cache_raw_image = input_dict['raw_images'][0][0]
        else:
            self.cache_raw_image = None
        del(input_dict['raw_images'])
        if len(input_dict['input_ids'].shape) == 1:
            input_dict['input_ids'] = input_dict['input_ids'].unsqueeze(0)
            input_dict['attention_mask'] = input_dict['attention_mask'].unsqueeze(0)
        if isinstance(input_dict['input_images'], list):
            input_dict['input_images'] = [item.to(self.dtype).to(self.device) if item is not None else item for item in input_dict['input_images']]
        else:
            input_dict['input_images'] = [input_dict['input_images'].to(self.dtype).to(self.device)] if input_dict['input_images'] is not None else [None]
        with torch.no_grad():
            text_out = self.generate(
                        input_ids = input_dict['input_ids'].to(self.device), # TODO unsqueeze is for bs==1
                        input_images=input_dict['input_images'] if input_dict['input_images'] is not None else [None],
                        attention_mask = input_dict['attention_mask'].to(self.device),
                        box = input_dict['box'] if 'box' in input_dict else None,
                        do_sample = True if temperature > 0 else False,
                        temperature=temperature,
                        max_new_tokens = max_new_tokens,
                        pad_token_id = self.tokenizer.pad_token_id,
                        return_dict_in_generate = True
                    )

        input_token_len = input_dict['input_ids'].shape[1]
        n_diff_input_output = (input_dict['input_ids'].to(self.device) != text_out.sequences[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        pred_out = self.tokenizer.batch_decode(text_out.sequences[:, input_token_len:], skip_special_tokens=False)
        
        # check if images are in the output, if so, decode them
        output_images = []
        if len(self.to_generate_images) and not avoid_image_gen:
            self.vision_generator.image_pipeline.to(self.device, self.dtype)
            negative_embeds = self.encode_img(torch.zeros(1, 3, self.vision_encoder.image_size, self.vision_encoder.image_size).to(device=self.device, dtype=self.dtype))[2]
            for to_gen_img in self.to_generate_images:
                out_img = self.vision_generator.image_pipeline(prompt_embeds=to_gen_img, negative_embeds=negative_embeds, guidance_scale=3, height=1024, width=1024, crop_info=[0, 0], original_size=[1024, 1024], num_inference_steps=100).image         
                output_images.append(out_img)
        return pred_out, output_images, text_out.sequences

    def calculate_options(self, input_dict, cot=False, further_instruct=False, temperature=0.2, max_new_tokens=128, guidance_scale=7.5, avoid_image_gen=False, likelihood_reduction='mean', **kwargs):
        
        assert len(input_dict['options']) == 1
        self.cache_images = None
        self.boi_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_BOI_TOKEN])[0]
        options = [s for s in input_dict['options'][0]]
        del(input_dict['options'])
        if self.sub_image_bind:
            self.cache_raw_image = input_dict['raw_images'][0][0]
        else:
            self.cache_raw_image = None
        del(input_dict['raw_images'])
        if len(input_dict['input_ids'].shape) == 1:
            input_dict['input_ids'] = input_dict['input_ids'].unsqueeze(0)
            input_dict['attention_mask'] = input_dict['attention_mask'].unsqueeze(0)
        if isinstance(input_dict['input_images'], list):
            input_dict['input_images'] = [item.to(self.dtype).to(self.device) if item is not None else item for item in input_dict['input_images']]
        else:
            input_dict['input_images'] = [input_dict['input_images'].to(self.dtype).to(self.device)] if input_dict['input_images'] is not None else [None]
        if cot:
            # need to first conduct the thinking
            with torch.no_grad():
                text_out = self.generate(
                            input_ids = input_dict['input_ids'].to(self.device), # TODO unsqueeze is for bs==1
                            input_images=input_dict['input_images'] if input_dict['input_images'] is not None else [None],
                            attention_mask = input_dict['attention_mask'].to(self.device),
                            box = input_dict['box'] if 'box' in input_dict else None,
                            do_sample = True if temperature > 0 else False,
                            temperature=temperature,
                            max_new_tokens = max_new_tokens,
                            pad_token_id = self.tokenizer.pad_token_id,
                            return_dict_in_generate = True
                        )
            thought_ids = text_out.sequences
            input_token_len = input_dict['input_ids'].shape[1]
            n_diff_input_output = (input_dict['input_ids'].to(self.device) != text_out.sequences[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            thought = self.tokenizer.batch_decode(text_out.sequences[:, input_token_len:], skip_special_tokens=False)[0]
            thought_ids = thought_ids.squeeze()
            if further_instruct:
                # need to instruct the model to select from options!
                options_instruct = 'Select from following options: ' + '; '.join(options) + '.'
                option_instruct_ids = self.tokenizer([options_instruct], return_tensors='pt')['input_ids'][:, 1:].squeeze()
                suffix = torch.cat([torch.tensor([733, 16289, 28793,  1824,   349,   574,  1480,  4372, 28804]), 
                                    option_instruct_ids, 
                                    torch.tensor([733, 28748, 16289, 28793])]).to(thought_ids.device)
            else:
                suffix = torch.tensor([  733, 16289, 28793,  1824,   349,   574,  1480,  4372, 28804,   733,
                                        28748, 16289, 28793]).to(thought_ids.device)
            eoc_indices = [-1] + torch.where(thought_ids == self.eoc_token_id)[0].tolist() + [thought_ids.shape[0]-1]
            input_ids = []
            for i in range(len(eoc_indices) - 1):
                input_ids.append(thought_ids[eoc_indices[i]+1:eoc_indices[i+1]+1])
                if i < len(eoc_indices) - 2:
                    if thought_ids[eoc_indices[i+1]+1].item() != self.input_img_id:
                        input_ids.append(torch.tensor([self.input_img_id]).to(thought_ids.device))
            input_ids.append(suffix)
            input_ids = torch.cat(input_ids).unsqueeze(0)
            new_thought, thought_boxes = process_thought(thought, mistral=True)
            if self.sub_image_bind:
                w, h = self.cache_raw_image.size
                new_sub_images = []
                for b in thought_boxes:
                    x_min, y_min, x_max, y_max = b
                    x_min = x_min*w
                    y_min = y_min*h
                    x_max = x_max*w
                    y_max = y_max*h
                    sub_image = self.cache_raw_image.crop((x_min, y_min, x_max, y_max))
                    new_sub_images.append(self.image_processor(resize_image_to_square(sub_image)).unsqueeze(0).to(dtype=self.dtype, device=input_ids.device))
                num_images = len(new_sub_images) + 1
                input_dict['input_images'] = [torch.cat(input_dict['input_images'] + new_sub_images, dim=0)]
                all_box = [torch.tensor([[0.0, 0.0, 1.0, 1.0]]*num_images, device=input_ids.device, dtype=input_dict['box'][0].dtype)]
            else: 
                all_box = input_dict['box'][0]
                all_box = [torch.cat([all_box, torch.tensor(thought_boxes, device=all_box.device, dtype=all_box.dtype)], dim=0)]
        else:
            input_ids = input_dict['input_ids']
            all_box = input_dict['box']
        
        # calculate the past qk for processing
        input_dict['input_ids'] = input_ids.to(self.device)
        input_dict['box'] = all_box
        input_dict['use_cache'] = True
        input_dict['attention_mask'] = torch.ones_like(input_dict['input_ids']).to(self.device)
        del(input_dict['labels'])
        question_output = self.forward(**input_dict)
        question_logits = question_output.logits

        # calculate the logit
        option_losses = []
        for opt in options:
            opt_ids = self.tokenizer([opt], return_tensors='pt')['input_ids'][:, 1:].to(self.device)
            opt_output = self.forward(input_ids = opt_ids, attention_mask=torch.ones_like(opt_ids), 
                                    past_key_values=question_output.past_key_values, use_cache=True)
            logits = torch.cat([question_logits[:, -1:], opt_output.logits[:, :-1]], 1)
            if likelihood_reduction == 'mean':
                loss_fct = CrossEntropyLoss()
                logits = logits.view(-1, self.config.vocab_size)
                labels = opt_ids.view(-1)
                loss = loss_fct(logits, labels)
            elif likelihood_reduction == 'sum':
                loss_fct = CrossEntropyLoss(reduction='none')
                logits = logits.view(-1, self.config.vocab_size)
                labels = opt_ids.view(-1)
                loss = loss_fct(logits, labels).sum()
            else:
                raise ValueError
            option_losses.append(loss)
        
        return torch.stack(option_losses).argmin().cpu().item(), thought if cot else None

    def load_state_dict_from_ckpt(self,ckpt_file):
        state_dict = torch.load(ckpt_file)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        new_state_dict = dict()
        for key in state_dict.keys():
            if 't2i_decoder_prompt' in key or 'llm_to_t2i_mapping' in key:
                new_key = 'behind_projector.'+key
            elif 'llama_model' in key:
                if 'lora' in key or 'modules_to_save' in key:
                    new_key = '.'.join(key.split('.')[4:])
                else:
                    new_key = '.'.join(key.split('.')[2:])
            new_state_dict[new_key] = state_dict[key]

        model_state_dict = self.state_dict()
        model_state_dict.update(new_state_dict)
        self.load_state_dict(model_state_dict)

    def load_state_dict_from_old_code_checkpoint(self,ckpt_file):
        state_dict = torch.load(ckpt_file)
        prefix_key = list(set(['.'.join(key.split('.')[:2]) for key in state_dict.keys()]))
        new_state_dict = dict()
        for key in state_dict.keys():
            if 't2i_decoder_prompt' in key or 'llm_to_t2i_mapping' in key:
                new_key = 'behind_projector.'+key
            elif 'llama_model' in key:
                new_key = '.'.join(key.split('.')[4:])
            elif 'vae' in key or 'unet' in key or 'sd_text_encoder' in key:
                new_key = 'vision_generator.'+key
                # check the unet condition conv_in layer
                if 'unet.conv_in.weight' == key:
                    num_channels = state_dict[key].shape[1]
                    if self.image_condition:
                        if self.config.vision_generator_cond_channels + 4 != num_channels:
                            continue
            elif 'Qformer' in key:
                new_key = 'front_mm_projector.'+'.'.join(key.split('.')[2:])
            elif 'query_tokens' in key or 'llama_proj' in key:
                new_key = '.'.join(key.split('.')[1:])
            elif 'ln_vision' in key:
                new_key = 'vit_ln.' + '.'.join(key.split('.')[1:])
            elif 'visual_encoder' in key:
                new_key = 'vision_encoder.'+ '.'.join(key.split('.')[2:])
            elif 'fc' in key:
                new_key = key
            else:
                raise ValueError('no support key from old code checkpoint')

            new_state_dict[new_key] = state_dict[key]

        self.load_state_dict(new_state_dict, strict=False)
AutoConfig.register("VolCanoMistral", VolCanoMistralConfig)
AutoModelForCausalLM.register(VolCanoMistralConfig, VolCanoMistralForCausalLM)
