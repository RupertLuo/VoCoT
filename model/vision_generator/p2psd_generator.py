import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel 
from diffusers import StableDiffusionPipeline, StableDiffusionInstructPix2PixPipeline
from constants import *
from transformers.deepspeed import deepspeed_config, is_deepspeed_zero3_enabled
from transformers import AutoConfig
from diffusers.models.vae import DiagonalGaussianDistribution
import torch.nn.functional as F
import random

class P2P_SDVisionGenerator(nn.Module):
    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        sd_model_name = config.vision_generator
        self.config = config

        self.sd_text_encoder = CLIPTextModel.from_pretrained(sd_model_name, subfolder="text_encoder")
        self.sd_tokenizer = CLIPTokenizer.from_pretrained(sd_model_name, subfolder="tokenizer")
        self.vae = AutoencoderKL.from_pretrained(sd_model_name, subfolder="vae")
        
        self.unet = UNet2DConditionModel.from_pretrained(sd_model_name, subfolder="unet")
        # Freeze vae and text_encoder, need to be edited here
        self.vae.requires_grad_(False)
        self.sd_text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)

        self.image_pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            sd_model_name,
            vae = self.vae,
            unet = self.unet,
            safety_checker = None,
        )

        # resize the conv_in layer
        cond_channels = getattr(config, 'vision_generator_cond_channels', 4)
        if cond_channels > 0:
            self.resize_conv_in(cond_channels + 4)

        self.noise_scheduler = self.image_pipeline.scheduler
        empty_text_feature = self.encode_caption('', self.sd_tokenizer.model_max_length, inference=True)
        self.register_buffer('empty_text_feature', empty_text_feature, persistent=False)
    
    def save_pretrained(self, path):
        self.image_pipeline.save_pretrained(path)
    
    def resize_conv_in(self, target_in_channels):
        # define the new conv_in layer
        original_in_channel = self.unet.config.in_channels
        if original_in_channel == target_in_channels:
            return True
        print('resize the conv_in channels from {} to {}'.format(original_in_channel, target_in_channels))
        old_conv_in = self.unet.conv_in
        self.unet.config.in_channels = target_in_channels
        self.unet.config['in_channels'] = target_in_channels

        new_conv_in = nn.Conv2d(in_channels=target_in_channels, out_channels=self.unet.conv_in.out_channels,
                                kernel_size=self.unet.conv_in.kernel_size, stride=self.unet.conv_in.stride,
                                padding=self.unet.conv_in.padding)
        new_conv_in.to(device=self.unet.conv_in.weight.device, dtype=self.unet.conv_in.weight.dtype)
        
        # init the new_conv_in to zero
        new_conv_in.weight.data.zero_()
        new_conv_in.bias.data.zero_()

        # copy the original weight to the new layer
        # numbers of channels to copy
        n = min(target_in_channels, original_in_channel)
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(old_conv_in.weight, modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    new_conv_in.weight.data[:, :n, :, :] = old_conv_in.weight.data[:, :n, :, :]
            with deepspeed.zero.GatheredParameters(old_conv_in.bias, modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    new_conv_in.bias.data = old_conv_in.bias.data
        else:
            new_conv_in.weight.data[:, :n, :, :] = old_conv_in.weight.data[:, :n, :, :]
            new_conv_in.bias.data = old_conv_in.bias.data
        
        # set the new conv_in layer
        self.unet.conv_in = new_conv_in
        return True

    def encode_caption(self, caption, length, inference=False):
        # add_special_tokens = False
        # if len(caption) == 0:
        add_special_tokens = True
        text_inputs = self.sd_tokenizer(
                caption,
                padding="max_length",
                max_length=length,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=add_special_tokens
            ).to(self.sd_text_encoder.device)
        # text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        prompt_embeds = self.sd_text_encoder(**text_inputs)[0]
        return prompt_embeds
    
    def forward(self, item_id, input_ids, output_images, output_cond_images, output_cond_img_mask, output_image_feature=None):
        if output_image_feature is not None:
            latents = DiagonalGaussianDistribution(output_image_feature).sample()
        else:
            if len(output_images.shape) == 3:
                output_images = output_images.unsqueeze(0)

            latents = self.vae.encode(output_images).latent_dist.sample()
        
        assert output_cond_images is not None, "the current model requires image as conditions"
        # mask the uncond (can be accelerated here TODO!)
        bs = input_ids.shape[0]
        random_probs = torch.rand(bs, device=input_ids.device)# [random.random() for i in range(bs)]
        prompt_mask = (random_probs < 0.1).unsqueeze(1).unsqueeze(2)

        # make the text condition
        encoder_hidden_states = self.sd_text_encoder(input_ids)[0]

        encoder_hidden_states = torch.where(prompt_mask, self.empty_text_feature.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype),
                                  encoder_hidden_states)

        image_cond_latents = self.vae.encode(output_cond_images).latent_dist.mode()
        random_image_mask = 1 - (
            (random_probs >= 0.05).to(dtype=image_cond_latents.dtype) * (random_probs < 0.15).to(dtype=image_cond_latents.dtype)
        )
        cond_img_mask = output_cond_img_mask.to(dtype=image_cond_latents.dtype) * random_image_mask.to(dtype=image_cond_latents.dtype)
        cond_img_mask = cond_img_mask.to(image_cond_latents.dtype).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        image_cond_latents = cond_img_mask*image_cond_latents
        
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
        timesteps = timesteps.long()

        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        target = noise

        # concatenate the image condition in the channels
        noisy_latents = torch.cat([noisy_latents, image_cond_latents], dim=1)
        model_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample


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
        return {"loss": loss}
    
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

if __name__=='__main__':
    model = P2P_SDVisionGenerator()