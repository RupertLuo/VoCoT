import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel 
from diffusers import StableDiffusionPipeline
from constants import *

class SDVisionGenerator(nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        sd_model_name = config.vision_generator

        self.sd_text_encoder = CLIPTextModel.from_pretrained(sd_model_name, subfolder="text_encoder")
        self.sd_tokenizer = CLIPTokenizer.from_pretrained(sd_model_name, subfolder="tokenizer")
        self.vae = AutoencoderKL.from_pretrained(sd_model_name, subfolder="vae")
        
        self.unet = UNet2DConditionModel.from_pretrained(sd_model_name, subfolder="unet")
        # Freeze vae and text_encoder
        self.vae.requires_grad_(False)
        self.sd_text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.image_pipeline = StableDiffusionPipeline.from_pretrained(
            sd_model_name,
            vae = self.vae,
            unet = self.unet,
            safety_checker = None,
        )

        self.noise_scheduler = self.image_pipeline.scheduler



        