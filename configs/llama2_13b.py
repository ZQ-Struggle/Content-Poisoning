import os

import torch

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    config.model_name = "llama2_13b"

    config.result_prefix = 'results/'

    config.tokenizer_paths=["./models/llama2-13b-tokenizer"]
    # config.tokenizer_paths=["meta-llama/Llama-2-13b-chat-hf"]
    config.model_paths=["./models/llama2-13b-chat-hf"]
    config.conversation_templates=['llama-2']
    config.model_kwargs = [{
        'device_map': 'auto'}]
    config.data_type = torch.float16
    config.devices = ['cuda']

    return config