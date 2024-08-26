import os

import torch

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    config.model_name = "llama2"

    config.result_prefix = 'results/'

    # config.tokenizer_paths=["/DIR/llama-2/llama/llama-2-7b-chat-hf"]
    # config.model_paths=["/DIR/llama-2/llama/llama-2-7b-chat-hf"]
    # config.conversation_templates=['llama-2']
    config.tokenizer_paths=["./models/llama2-tokenizer"]
    config.model_paths=["./models/llama-2-7b-chat-hf/"]
    config.conversation_templates=['llama-2']
    config.data_type = torch.float16
    config.model_kwargs = [{
        'device_map': 'auto'}]

    return config