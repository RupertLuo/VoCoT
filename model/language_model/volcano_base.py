from typing import List, Optional, Tuple, Union
from xml.dom.expatbuilder import parseString
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                        LlamaConfig, LlamaModel, LlamaForCausalLM, CLIPImageProcessor
# from .modeling_llama import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers import MistralForCausalLM, MistralConfig

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput

from ..vision_encoder.builder import build_vision_encoder
from ..vision_generator.builder import build_vision_generator
from ..front_projector.builder import build_front_projector
from ..behind_projector.builder import build_behind_projector
from transformers import GenerationMixin

from PIL import Image
from constants import *
from abc import ABC
import random
from utils.util import rank_0_print
import torch.nn.functional as F
# from diffusers.models.vae import DiagonalGaussianDistribution
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


class VolCanoConfig(LlamaConfig):
    model_type = "VolCano"


class VolCanoLlamaModel(LlamaModel):
    config_class = VolCanoConfig

    def __init__(self, config: LlamaConfig):
        super(VolCanoLlamaModel, self).__init__(config)
        

class VolCanoMetaForCausalLM(ABC):

    def init_tokenizer(self, tokenizer):
        self.num_new_tokens1 = tokenizer.add_special_tokens(
            {   
                # "pad_token": DEFAULT_PAD_TOKEN,
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
                "pad_token": DEFAULT_PAD_TOKEN
            }
        )
        # append image related tokens
        num_new_img_tokens = tokenizer.add_special_tokens(
                {   
                    "additional_special_tokens": [DEFAULT_IMG_TOKEN, DEFAULT_BOI_TOKEN, DEFAULT_EOI_TOKEN]
                }
            )
        self.num_new_tokens = self.num_new_tokens1 + num_new_img_tokens
        if self.num_new_tokens > 0:
            print('resize token embedding to {}'.format(len(tokenizer)))
            self.resize_token_embeddings(len(tokenizer))
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-self.num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-self.num_new_tokens:] = input_embeddings_avg
            output_embeddings[-self.num_new_tokens:] = output_embeddings_avg
        
        self.input_img_id = tokenizer.convert_tokens_to_ids('<ImageHere>')

        input_embed_grad_mask = torch.ones_like(self.get_input_embeddings().weight.data)
        output_embed_grad_mask = torch.ones_like(self.get_output_embeddings().weight.data)
        input_embed_grad_mask[:-self.num_new_tokens] = 0
        output_embed_grad_mask[:-self.num_new_tokens] = 0
        self.register_buffer("input_embed_grad_mask", input_embed_grad_mask, persistent=False)
        self.register_buffer("output_embed_grad_mask", output_embed_grad_mask, persistent=False)
    
    def init_tokenizer_loc(self, tokenizer):
        # append image related tokens
        new_coor_tokens = [DEFAULT_BOC_TOKEN, DEFAULT_EOC_TOKEN]
        new_phrase_tokens = [DEFAULT_BOP_TOKEN, DEFAULT_EOP_TOKEN]

        num_new_img_tokens = tokenizer.add_special_tokens(
                {   
                    "additional_special_tokens": [DEFAULT_GRD_TOKEN] + new_coor_tokens + new_phrase_tokens + ALL_LOC_TOKENS + [DEFAULT_SEP_TOKEN]
                }
            )
        # self.num_new_tokens = num_new_img_tokens
        if num_new_img_tokens > 0:
            print('resize token embedding to {}'.format(len(tokenizer)))
            self.resize_token_embeddings(len(tokenizer))
        
        self.input_img_id = tokenizer.convert_tokens_to_ids('<ImageHere>')
    
    def init_tokenizer_grd(self, tokenizer):
        # append image related tokens
        new_coor_tokens = [DEFAULT_BOC_TOKEN, DEFAULT_EOC_TOKEN]

        num_new_img_tokens = tokenizer.add_special_tokens(
                {   
                    "additional_special_tokens": [DEFAULT_GRD_TOKEN] + new_coor_tokens
                }
            )
        # self.num_new_tokens = num_new_img_tokens
        if num_new_img_tokens > 0:
            print('resize token embedding to {}'.format(len(tokenizer)))
            self.resize_token_embeddings(len(tokenizer))
        
        # <ImageHere> should not be included in the vocabulary
        tokenizer.add_special_tokens(
                {   
                    "additional_special_tokens": [DEFAULT_IMG_TOKEN]
                }
            )
        self.input_img_id = tokenizer.convert_tokens_to_ids('<ImageHere>')
    
    def reinit_partial_embeddings(self, valid_size=None):
        if valid_size is None:
            return
        print('reducing the embedding size to {}'.format(valid_size))
        self.resize_token_embeddings(valid_size)
        print('resizing the embedding size to {}'.format(len(self.tokenizer)))
        self.resize_token_embeddings(len(self.tokenizer))
    
    def init_vision_model(self):
        config = self.config
        if hasattr(config, "vision_encoder"):
            print('Create VIT')
            self.vision_encoder = build_vision_encoder(config, delay_load=False)
            if 'emu' in getattr(config, 'vision_encoder', None):
                self.n_query = getattr(config, 'num_image_token', 64)
                print('using {} tokens to represent images'.format(self.n_query))
            if 'eva' in getattr(config, 'vision_encoder', None):
                self.vit_ln = LayerNorm(self.vision_encoder.num_features)
                for name, param in self.vit_ln.named_parameters():
                    param.requires_grad = False
                self.vit_ln = self.vit_ln.eval()
                self.vit_ln.train = disabled_train
            if 'openai' in getattr(config, 'vision_encoder', None):
                self.n_query = int((self.vision_encoder.config.image_size / self.vision_encoder.config.patch_size)**2)
            config.mm_hidden_size = self.vision_encoder.num_features
            print('End Create VIT')
        
        if hasattr(config, "front_projector"):
            print('Create Front Projector')
            if getattr(config, 'front_projector_type', 'q_former') == 'q_former':
                self.front_mm_projector, self.query_tokens = build_front_projector(config, delay_load=False, visual_encoder = self.vision_encoder, num_query_token = config.num_query_token)
                self.llama_proj = nn.Linear( self.front_mm_projector.config.hidden_size, config.hidden_size )
            else:
                self.front_mm_projector = build_front_projector(config, delay_load=False)
            print('End Create Front Projector')

        if hasattr(config, "vision_generator"):
            if (not getattr(config, 'avoid_generator', False)):
                print('Create Vision Generator')
                self.vision_generator = build_vision_generator(config)
                self.image_condition = getattr(config, 'vision_generator_type', 'SD') == 'P2P_SD'
                self.sd_add_args = getattr(config, 'vision_generator_type', 'SD') == 'Emu2_SD'
                sd_hidden_size = self.vision_generator.unet.config.cross_attention_dim
                self.noise_scheduler = self.vision_generator.scheduler
                assert sd_hidden_size == config.mm_hidden_size, "the emu-2-based model must follow autoencoder structure!"
                config.sd_hidden_size = sd_hidden_size
                # self.fc = nn.Linear(config.hidden_size, sd_hidden_size)
                print('End Behind Generator')
            else:
                config.sd_hidden_size = config.mm_hidden_size
                self.vision_generator = None

        if hasattr(config, "behind_projector"):
            print('Create Behind Projector')
            self.behind_projector = build_behind_projector(config, vision_generator = self.vision_generator)
            print('End Behind Projector')

        if hasattr(config, "vision_generator"):
            self.diffusion_loss = getattr(config, "compute_diffusion_loss", False)
            pass
            # self.register_buffer('zero_img_feature', zero_img_feature, persistent=False)
        else:
            self.diffusion_loss = False

    def init_vision_generator(self):
        config = self.config
        if hasattr(config, "vision_generator"):
            if (not getattr(config, 'avoid_generator', False)):
                print('Create Vision Generator')
                self.vision_generator = build_vision_generator(config)
                self.image_condition = getattr(config, 'vision_generator_type', 'SD') == 'P2P_SD'
                self.sd_add_args = getattr(config, 'vision_generator_type', 'SD') == 'Emu2_SD'
                sd_hidden_size = self.vision_generator.unet.config.cross_attention_dim
                self.noise_scheduler = self.vision_generator.scheduler
                assert sd_hidden_size == config.mm_hidden_size, "the emu-2-based model must follow autoencoder structure!"
                config.sd_hidden_size = sd_hidden_size
                # self.fc = nn.Linear(config.hidden_size, sd_hidden_size)
                print('End Behind Generator')
            else:
                config.sd_hidden_size = config.mm_hidden_size
                self.vision_generator = None
        if hasattr(config, "vision_generator"):
            self.diffusion_loss = getattr(config, "compute_diffusion_loss", False)
            pass
            # self.register_buffer('zero_img_feature', zero_img_feature, persistent=False)
        else:
            self.diffusion_loss = False
    
    def get_model(self):
        return self.model

    def box_align(self, image, bboxes):
        # image: [n_patch * n_patch, hidden_size]
        # bboxes: [n_box, 4]
        feat_len, hidden_size = image.shape
        num_patches = int(feat_len**0.5)
        image_feat_2d = image.reshape(num_patches, num_patches, -1)
        num_boxes = bboxes.shape[0]
        bboxes_index = num_patches * bboxes
        x_min = torch.floor(bboxes_index[:, 0]).long()
        y_min = torch.floor(bboxes_index[:, 1]).long()
        x_max = torch.ceil(bboxes_index[:, 2]).long()
        y_max = torch.ceil(bboxes_index[:, 3]).long()
        box_feat = []
        for i in range(num_boxes):
            box_feat.append(image_feat_2d[x_min[i]:x_max[i], y_min[i]:y_max[i]].reshape(-1, hidden_size).contiguous())
        return box_feat
    
    def encode_img(self, image):
        with torch.no_grad():
            # no_grad since the visual encoder will not be updated in any stages
            # visual encoder
            if getattr(self.config, 'vision_encoder', 'eva_vit_emu') == 'eva_vit_emu':
                # using emu2-based encoders, need to perform avg-pooling!
                n_query = self.n_query

                image_embeds = self.vision_encoder(image)
                image_embeds = image_embeds[:, 1:, :]
                b, n, c = image_embeds.shape
                sqrt_n = int(n**0.5)
                image_embeds = image_embeds.permute(0, 2, 1).view(b, c, sqrt_n, sqrt_n)

                stride = int(sqrt_n // (n_query ** 0.5))
                image_embeds = F.avg_pool2d(image_embeds, kernel_size=(stride, stride), stride=stride)
                image_embeds = image_embeds.view(b, c, -1).permute(0, 2, 1).contiguous()
            else:
                image_embeds = self.vision_encoder(image)
            
        # visual connection module
        # no_grad here since connection module requires grads
        image_encoder_type = getattr(self.config, 'front_projector_type', 'q_former')
        if  image_encoder_type == 'q_former':
            # q-former based visual representations
            image_embeds = self.vit_ln(image_embeds)

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image_embeds.device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.front_mm_projector.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
            return inputs_llama, atts_llama, query_output.last_hidden_state
        elif image_encoder_type == 'linear' or 'mlp' in image_encoder_type:
            # linear mapping
            inputs_llama = self.front_mm_projector(image_embeds)
            atts_llama = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
            return inputs_llama, atts_llama, image_embeds
        else:
            raise NotImplementedError
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_label_masks=None, inputs_embeds=None, box=None
    ):
        vision_tower = self.vision_encoder
        current_device = input_ids.device if input_ids is not None else inputs_embeds.device
        current_dtype = self.dtype
        if input_ids is None and inputs_embeds is not None:
            # the branch for image geneartion in emu-based models
            target_shape = past_key_values[-1][-1].shape[-2] + inputs_embeds.shape[1]# 1 + self.n_query # <Img> and 64 image tokens
            attention_mask = torch.cat((attention_mask, torch.ones(
                (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )), dim=1)
            sentence_length = torch.sum(attention_mask, dim=1).item()
            position_ids = torch.arange(sentence_length-(inputs_embeds.shape[1]), sentence_length).unsqueeze(0).to(attention_mask.device)
            return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels, None, None
        elif past_key_values is not None and images is None and input_ids.shape[1] != 1:
            target_shape = past_key_values[-1][-1].shape[-2] + input_ids.shape[1]
            attention_mask = torch.cat((attention_mask, torch.ones(
                (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                dtype=attention_mask.dtype,
                device=attention_mask.device
            )), dim=1)
            sentence_length = torch.sum(attention_mask, dim=1).item()
            position_ids = torch.arange(sentence_length-(input_ids.shape[1]), sentence_length).unsqueeze(0).to(attention_mask.device)
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None
        elif vision_tower is None or images is None or input_ids.shape[1] == 1 or (input_ids.shape[1] == len(ALL_IMG_TOKENS) and input_ids[:,0]==self.output_img_id):
            # if last generation is [IMG0] or a new text
            if past_key_values is not None and vision_tower is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, None

        # encode the images in a dense manner
        all_valid_images = [item for item in images if item is not None]
        all_valid_image_size = [item.shape[0] for item in all_valid_images]
        if len(all_valid_images) > 0:
            # there is at least one image in the batch
            all_valid_images = torch.cat(all_valid_images, dim=0)
            all_valid_image_feature, all_valid_images_attention_mask, all_valid_vit_features = self.encode_img(all_valid_images)
            all_valid_image_feature = all_valid_image_feature.split(all_valid_image_size)
            all_valid_images_attention_mask = all_valid_images_attention_mask.split(all_valid_image_size)
            all_valid_vit_features = all_valid_vit_features.split(all_valid_image_size)
            zero_image_emb = None
        else:
            # all-text batch detected, need a zero_emb to keep the gradient on visual encoder + mm_projector consistent on all gpus!
            zero_image_emb = self.encode_img(torch.zeros(1, 3, self.vision_encoder.image_size, self.vision_encoder.image_size).to(device=self.device, dtype=self.dtype))[0]

        image_features = []
        visual_labels = []
        visual_label_masks = []
        images_attention_masks = []
        valid_image_index = 0
        if box is None:
            box = [None for _ in range(input_ids.shape[0])]
        for i,image in enumerate(images):
            if image is not None:
                # image_feature, images_attention_mask, vit_features = self.encode_img(image)
                image_feature, images_attention_mask, vit_features = all_valid_image_feature[valid_image_index], all_valid_images_attention_mask[valid_image_index], all_valid_vit_features[valid_image_index]
                current_box_info = box[i]
                if current_box_info is not None and image_feature.shape[0] == 1 and current_box_info.shape[0] > 1:
                    # need box alignment
                    aligned_box_feat = self.box_align(image_feature[0], current_box_info)
                    for box_feat in aligned_box_feat:
                        box_feat_len = box_feat.shape[0]
                        image_features.append(box_feat)
                        # for representation boxes, no need for visual label prediction
                        visual_labels.append(torch.zeros(box_feat_len, self.config.mm_hidden_size).to(current_dtype).to(current_device))
                        visual_label_masks.append(torch.zeros(box_feat_len).to(device=current_device, dtype=current_dtype))
                        images_attention_masks.append(torch.ones(box_feat_len, dtype=torch.long).to(current_device))
                else:
                    tmp_num_images = image_feature.shape[0]
                    image_features.extend(split_tensor_func(image_feature))
                    images_attention_masks.extend(split_tensor_func(images_attention_mask))
                    visual_labels.extend(split_tensor_func(vit_features))
                    num_img_sample = image_feature.shape[0]
                    if image_label_masks is None or image_label_masks[i] is None:
                        tmp_mask = torch.zeros(num_img_sample, self.n_query).to(device=current_device, dtype=current_dtype)
                    else:
                        tmp_mask = image_label_masks[i].unsqueeze(1).to(current_dtype) * torch.ones(image_feature.shape[:2], device=current_device, dtype=current_dtype)
                    visual_label_masks.extend(split_tensor_func(tmp_mask))
                valid_image_index += 1
            else:
                if zero_image_emb is None:
                    # there is at least one image in the batch
                    image_features.append(torch.zeros(self.n_query, self.config.hidden_size).to(current_dtype).to(current_device))
                else:
                    # no images in the batch
                    image_features.append(zero_image_emb.squeeze(0))
                visual_labels.append(torch.zeros(self.n_query, self.config.mm_hidden_size).to(current_dtype).to(current_device))
                visual_label_masks.append(torch.zeros(self.n_query).to(device=current_device, dtype=current_dtype))
        # image_features = torch.cat(image_features, dim=0)
        # visual_labels = torch.cat(visual_labels, dim=0)
        # visual_label_masks = torch.cat(visual_label_masks, dim=0)
        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        new_visual_labels = [] # the visual labels are the regression targets
        new_visual_label_masks = [] # the visual label masks indicating if loss is required
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == self.input_img_id).sum()
            # process the sequence without images
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                new_visual_labels.append(torch.zeros(cur_input_embeds.shape[0], self.config.mm_hidden_size).to(dtype=current_dtype, device=current_device))
                new_visual_label_masks.append(torch.zeros(cur_input_embeds.shape[0]).to(dtype=current_dtype, device=current_device))
                cur_image_idx += 1
                continue

            # split the sequence by <img>
            image_token_indices = [-1] + torch.where(cur_input_ids == self.input_img_id)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            # prepare the input
            cur_new_input_embeds = []
            cur_new_labels = []
            cur_new_visual_labels = []
            cur_new_visual_label_masks = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                # set the image labels of the text parts to zero
                cur_new_visual_labels.append(torch.zeros(cur_input_embeds_no_im[i].shape[0], self.config.mm_hidden_size).to(dtype=current_dtype, device=current_device))
                cur_new_visual_label_masks.append(torch.zeros(cur_input_embeds_no_im[i].shape[0]).to(dtype=current_dtype, device=current_device))

                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_visual_labels = visual_labels[cur_image_idx]
                    cur_visual_label_masks = visual_label_masks[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    cur_new_visual_labels.append(cur_visual_labels)
                    cur_new_visual_label_masks.append(cur_visual_label_masks)

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            cur_new_visual_labels = torch.cat(cur_new_visual_labels)
            cur_new_visual_label_masks = torch.cat(cur_new_visual_label_masks)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            new_visual_labels.append(cur_new_visual_labels)
            new_visual_label_masks.append(cur_new_visual_label_masks)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
            new_visual_labels = [x[:tokenizer_model_max_length] for x in new_visual_labels]
            new_visual_label_masks = [x[:tokenizer_model_max_length] for x in new_visual_label_masks]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_visual_labels_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_visual_label_masks_padded = torch.full((batch_size, max_len), 0, dtype=current_dtype, device=current_device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels, cur_new_visual_labels, cur_new_visual_label_masks) in enumerate(zip(new_input_embeds, new_labels, new_visual_labels, new_visual_label_masks)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                new_visual_labels_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_visual_labels.shape[1]), dtype=cur_new_visual_labels.dtype, device=cur_new_visual_labels.device),
                    cur_new_visual_labels
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    new_visual_label_masks_padded[i, -cur_len:] = cur_new_visual_label_masks
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                new_visual_labels_padded.append(torch.cat((
                    cur_new_visual_labels,
                    torch.zeros((max_len - cur_len, cur_new_visual_labels.shape[1]), dtype=cur_new_visual_labels.dtype, device=cur_new_visual_labels.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    new_visual_label_masks_padded[i, :cur_len] = cur_new_visual_label_masks
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        new_visual_labels = torch.stack(new_visual_labels_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        
        new_visual_label_masks = new_visual_label_masks_padded

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, new_visual_labels, new_visual_label_masks