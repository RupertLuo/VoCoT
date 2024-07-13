import torch
import os


IMG_TOKEN_NUM = 8
ALL_IMG_TOKENS = [f"[IMG{i}]" for i in range(IMG_TOKEN_NUM)]
ALL_IMG_TOKENS_STR = '<ImageHere>' # "<Img><ImageHere></Img>"# "".join(ALL_IMG_TOKENS)

# Location related tokens
LOC_TOKEN_NUM = 256
ALL_LOC_TOKENS = ["[LOC{}]".format(i+1) for i in range(LOC_TOKEN_NUM)]

USE_PREFIX_TUNING = False
USE_LORA = False
USE_CFG = True
IGNORE_TOKEN_ID = -100
IMAGE_TOKEN_INDEX = -200


PRECISION = torch.bfloat16
TRAINABLE_PRECISION = torch.float32

IGNORE_INDEX = -100
DEFAULT_GRD_TOKEN = "<grounding>"
DEFAULT_BOP_TOKEN = "" # begin of phrase modified from 3/11
DEFAULT_EOP_TOKEN = "" # end of phrase modified from 3/11
DEFAULT_BOC_TOKEN = "<coor>" # begin of coordinates
DEFAULT_EOC_TOKEN = "</coor>" # end of coordinates
DEFAULT_SEP_TOKEN =  " and"# "<delim>" modified from 3/11
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

# begin of image
DEFAULT_BOI_TOKEN = "<Img>"
# end of image
DEFAULT_EOI_TOKEN = '</Img>'
# default image token
DEFAULT_IMG_TOKEN = '<ImageHere>'
COT_ACTIVATION = 'Answer the question and include the reasoning proess. Locate key objects and provide bounding boxes in your thoughts.'
COT_ACTIVATION_TXT = 'Answer the question and include the reasoning proess.'