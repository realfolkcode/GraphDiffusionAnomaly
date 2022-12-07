import os
import time
import pickle
import math
import torch

from .utils.logger import Logger, set_log, start_log, train_log, sample_log, check_log
from .utils.loader import load_ckpt, load_data, load_seed, load_device, load_model_from_ckpt, \
                         load_ema_from_ckpt, load_sampling_fn, load_eval_settings
from .utils.graph_utils import adjs_to_graphs, init_flags, quantize, node_flags
from .utils.plot import save_graph_list, plot_graphs_list
from .evaluation.stats import eval_graph_list


# -------- Sampler for generic graph generation tasks --------
class Sampler(object):
    def __init__(self, config, num_sampling_rounds, train_loader):
        super(Sampler, self).__init__()

        self.config = config
        self.device = load_device()
        self.num_sampling_rounds = num_sampling_rounds
        self.train_loader = train_loader

    def sample(self):
        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict['config']

        load_seed(self.configt.seed)

        self.log_folder_name, self.log_dir, _ = set_log(self.configt, is_train=False)
        self.log_name = f"{self.config.ckpt}-sample"
        logger = Logger(str(os.path.join(self.log_dir, f'{self.log_name}.log')), mode='a')

        if not check_log(self.log_folder_name, self.log_name):
            logger.log(f'{self.log_name}')
            start_log(logger, self.configt)
            train_log(logger, self.configt)
        sample_log(logger, self.config)

        # -------- Load models --------
        self.model_x = load_model_from_ckpt(self.ckpt_dict['params_x'], self.ckpt_dict['x_state_dict'], self.device)
        self.model_adj = load_model_from_ckpt(self.ckpt_dict['params_adj'], self.ckpt_dict['adj_state_dict'], self.device)

        if self.config.sample.use_ema:
            self.ema_x = load_ema_from_ckpt(self.model_x, self.ckpt_dict['ema_x'], self.configt.train.ema)
            self.ema_adj = load_ema_from_ckpt(self.model_adj, self.ckpt_dict['ema_adj'], self.configt.train.ema)
            
            self.ema_x.copy_to(self.model_x.parameters())
            self.ema_adj.copy_to(self.model_adj.parameters())

        self.sampling_fn = load_sampling_fn(self.configt, self.config.sampler, self.config.sample, self.device)

        # -------- Generate samples --------
        logger.log(f'GEN SEED: {self.config.sample.seed}')
        load_seed(self.config.sample.seed)

        num_sampling_rounds = self.num_sampling_rounds
        gen_graph_list = []
        for r in range(num_sampling_rounds):
            t_start = time.time()

            _, train_adj = next(iter(self.train_loader))
            device_id = f'cuda:{self.device[0]}' if isinstance(self.device, list) else self.device
            train_adj = train_adj.to(device_id)
            self.init_flags = node_flags(train_adj)

            x, adj, _ = self.sampling_fn(self.model_x, self.model_adj, self.init_flags)

            logger.log(f"Round {r} : {time.time()-t_start:.2f}s")

            samples_int = quantize(adj)
            gen_graph_list.extend(adjs_to_graphs(samples_int, True))

        # -------- Save samples --------
        save_dir = save_graph_list(self.log_folder_name, self.log_name, gen_graph_list)
        with open(save_dir, 'rb') as f:
            sample_graph_list = pickle.load(f)
        plot_graphs_list(graphs=sample_graph_list, title=f'{self.config.ckpt}', max_num=16, save_dir=self.log_folder_name)
