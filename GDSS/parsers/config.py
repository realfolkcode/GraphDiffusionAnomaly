import yaml
from easydict import EasyDict as edict


def get_config(config_path, seed):
    config = edict(yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))
    config.seed = seed

    return config