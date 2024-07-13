from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator

import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers.utils import logging

USE_LORA = False
if not USE_LORA: 
    from .modeling_qwen import QWenModel, QWenLMHeadModel
else:
    from .modeling_qwen_lora import QWenModel, QWenLMHeadModel

SUPPORT_CUDA = torch.cuda.is_available()
SUPPORT_BF16 = SUPPORT_CUDA and torch.cuda.is_bf16_supported()
SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7
logger = logging.get_logger(__name__)


class MonkeyModel(QWenModel):
    _use_lora_by_hand = USE_LORA

    def __init__(self, config):
        super().__init__(config)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
    ):
        if past_key_values is None and torch.any(input_ids == self.config.visual['image_start_id']):
            bos_pos = torch.where(input_ids == self.config.visual['image_start_id'])
            eos_pos = torch.where(input_ids == self.config.visual['image_end_id'])
            assert (bos_pos[0] == eos_pos[0]).all()
            img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)

            if images is None:
                image_paths = []
                for pos_i, a, b in img_pos:
                    img_path = input_ids[pos_i][a + 1 : b - 1].tolist()
                    img_path = img_path[ : img_path.index(self.config.visual['image_pad_id'])]
                    image_paths.append(bytes(img_path).decode('utf-8'))
                windows, images_448 = self.visual.encode(image_paths=image_paths)
            else:
                image_idxs = []
                for pos_i, a, b in img_pos:
                    img_idx = input_ids[pos_i][a + 1 : b - 1].tolist()
                    img_idx = img_idx[ : img_idx.index(self.config.visual['image_pad_id'])]
                    image_idxs.append(bytes(img_idx).decode('utf-8'))
                assert image_idxs == sorted(image_idxs)
                windows, images_448 = self.visual.encode(images=images)

            # modify for dynamic resolution
            patch_list = []
            col_shift = self.config.visual['lora_cols_num']
            for row_idx, col in enumerate(windows):
                for col_idx, image_patch in enumerate(col):
                    lora_idx = row_idx * col_shift + col_idx
                    # print("LORA: ", lora_idx)
                    patch_list.append(self.visual(image_patch, idx=lora_idx))
                    lora_idx += 1
            
            global_feat = self.visual(images_448)                   # batch x 256               x 4096
            local_feat = torch.cat(patch_list, dim=1)               # batch x n_window*256      x 4096
            images = torch.cat([local_feat, global_feat], dim=1)    # batch x (n_window+1)*256  x 4096
        else:
            images = None

        return super().forward(
            input_ids,
            past_key_values,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            images,
        )


class MonkeyLMHeadModel(QWenLMHeadModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.rotary_emb\.inv_freq"]
    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.attn\.masked_bias"]
    _use_lora_by_hand = USE_LORA

    def __init__(self, config):
        super().__init__(config)
        assert (
            config.bf16 + config.fp16 + config.fp32 <= 1
        ), "Only one of \"bf16\", \"fp16\", \"fp32\" can be true"

        autoset_precision = config.bf16 + config.fp16 + config.fp32 == 0

        if autoset_precision:
            if SUPPORT_BF16:
                logger.warn(
                    "The model is automatically converting to bf16 for faster inference. "
                    "If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to \"AutoModelForCausalLM.from_pretrained\"."
                )
                config.bf16 = True
            elif SUPPORT_FP16:
                logger.warn(
                    "The model is automatically converting to fp16 for faster inference. "
                    "If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to \"AutoModelForCausalLM.from_pretrained\"."
                )
                config.fp16 = True
            else:
                config.fp32 = True

        if config.bf16 and SUPPORT_CUDA and not SUPPORT_BF16:
            logger.warn("Your device does NOT seem to support bf16, you can switch to fp16 or fp32 by by passing fp16/fp32=True in \"AutoModelForCausalLM.from_pretrained\".")
        if config.fp16 and SUPPORT_CUDA and not SUPPORT_FP16:
            logger.warn("Your device does NOT support faster inference with fp16, please switch to fp32 which is likely to be faster")
        if config.fp32:
            if SUPPORT_BF16:
                logger.warn("Your device support faster inference by passing bf16=True in \"AutoModelForCausalLM.from_pretrained\".")
            elif SUPPORT_FP16:
                logger.warn("Your device support faster inference by passing fp16=True in \"AutoModelForCausalLM.from_pretrained\".")

        assert config.visual['lora_cols_num'] * config.visual['lora_rows_num'] \
            == config.visual['lora_repeat_num'], "Visual-Lora shape must be match!"

        self.transformer = MonkeyModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.bf16:
            self.transformer.bfloat16()
            self.lm_head.bfloat16()
        if config.fp16:
            self.transformer.half()
            self.lm_head.half()
        self.post_init()

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs