from .sd_generator import SDVisionGenerator
from .p2psd_generator import P2P_SDVisionGenerator
from .emu_generator import EmuVisualGenerationPipeline, EmuSDVisionGenerator
import torch

def build_vision_generator(config):
    # distinguish the generator type defaulted to standard Stable Diffusion
    generator_type = getattr(config, 'vision_generator_type', 'SD')
    if generator_type == 'SD':
        return SDVisionGenerator(config)
    elif generator_type == 'P2P_SD':
        # using the architecture of instruct pix2pix
        return P2P_SDVisionGenerator(config)
    elif generator_type == 'Emu2_SD':
        return EmuSDVisionGenerator(config)
        # pipe = EmuVisualGenerationPipeline.from_pretrained(config.vision_generator, 
        #                                                    safety_checker=None,
        #                                                    use_safetensors=True, variant='bf16',
        #                                                    torch_dtype=torch.bfloat16)
        return pipe
    else:
        raise ValueError