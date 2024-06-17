import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    config.model_name = "vicuna_13b"

    # tokenizers
    config.tokenizer_paths=['./models/vicuna-13b-v1.5-16k']
    config.tokenizer_kwargs=[{"use_fast": False}]
    config.model_paths=['./models/vicuna-13b-v1.5-16k']
    config.model_kwargs = [{
        'device_map': 'auto'}]
    return config