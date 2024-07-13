import torch
import torch.nn as nn

class BehindProjector(nn.Module):
    def __init__(self,config, vision_generator = None):
        super().__init__()
        sd_hidden_size = config.t2i_mapping_hidden_size # vision_generator.sd_text_encoder.config.hidden_size
        self.t2i_decoder_prompt = torch.nn.Parameter(torch.randn((1,77, sd_hidden_size)))
        self.llm_to_t2i_mapping = nn.Transformer(batch_first=True, 
                                                 norm_first=True, 
                                                 d_model = sd_hidden_size, 
                                                 num_encoder_layers=4, 
                                                 num_decoder_layers=4, 
                                                 dim_feedforward=sd_hidden_size*4, 
                                                 dropout=0.0)
        


def build_behind_projector(config, **kwargs):
    behind_projector_type = getattr(config, 'behind_projector', 'transformer')
    if behind_projector_type == 'transformer':
        return BehindProjector(config, kwargs['vision_generator'])
    elif behind_projector_type == 'linear':
        return nn.Linear(config.hidden_size, config.sd_hidden_size)
    else:
        raise NotImplementedError


