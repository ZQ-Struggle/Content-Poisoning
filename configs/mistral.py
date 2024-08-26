import os

import torch

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/'
    config.model_name = "mistral"


    # config.tokenizer_paths=["./models/Mistral-7B-Instruct-v0.2/"]
    config.tokenizer_paths=["./models/mistral-tokenizer"]
    config.model_paths=["./models/Mistral-7B-Instruct-v0.2/"]
    config.conversation_templates=['llama-2']
    config.data_type = torch.float16
    config.model_kwargs = [{
        'device_map': 'auto'}]

    return config