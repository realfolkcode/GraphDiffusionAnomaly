import yaml
from easydict import EasyDict as edict
from copy import deepcopy


def get_config(config_path, seed):
    config = edict(yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))
    config.seed = seed

    return config


def save_config(config, config_path):
    simple_config = deepcopy(config)
    simple_config = dict(simple_config)
    simple_config['data'] = dict(simple_config['data'])
    simple_config['sde']['x'] = dict(simple_config['sde']['x'])
    simple_config['sde']['adj'] = dict(simple_config['sde']['adj'] )
    simple_config['sde'] = dict(simple_config['sde'])
    simple_config['model'] = dict(simple_config['model'])
    simple_config['train'] = dict(simple_config['train'])
    simple_config['sampler'] = dict(simple_config['sampler'])
    simple_config['sample'] = dict(simple_config['sample'])
    with open(config_path, 'w') as f:
        yaml.dump(simple_config, f)