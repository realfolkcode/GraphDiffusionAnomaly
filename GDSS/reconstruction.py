import torch

from GDSS.utils.loader import load_device, load_seed, load_model_from_ckpt, \
                              load_ema_from_ckpt, load_ckpt, load_sampling_fn, \
                              load_batch, load_sde
from GDSS.utils.graph_utils import node_flags, gen_noise, mask_x, mask_adjs, \
                                   quantize


class Reconstructor(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.device = load_device()

        # -------- Load checkpoint --------
        self.ckpt_dict = load_ckpt(self.config, self.device)
        self.configt = self.ckpt_dict['config']

        load_seed(self.configt.seed)

        # -------- Load models --------
        self.model_x = load_model_from_ckpt(self.ckpt_dict['params_x'], self.ckpt_dict['x_state_dict'], self.device)
        self.model_adj = load_model_from_ckpt(self.ckpt_dict['params_adj'], self.ckpt_dict['adj_state_dict'], self.device)

        if self.config.sample.use_ema:
            self.ema_x = load_ema_from_ckpt(self.model_x, self.ckpt_dict['ema_x'], self.configt.train.ema)
            self.ema_adj = load_ema_from_ckpt(self.model_adj, self.ckpt_dict['ema_adj'], self.configt.train.ema)
            
            self.ema_x.copy_to(self.model_x.parameters())
            self.ema_adj.copy_to(self.model_adj.parameters())

        self.sampling_fn = load_sampling_fn(self.config, self.config.sampler, self.config.sample, self.device)

        self.sde_x = load_sde(config.sde.x)
        self.sde_adj = load_sde(config.sde.adj)
    

    def perturb(self, x, adj):
        t = torch.ones(adj.shape[0], device=adj.device) * self.sde_adj.T
        flags = node_flags(adj)

        z_x = gen_noise(x, flags, sym=False)
        mean_x, std_x = self.sde_x.marginal_prob(x, t)
        perturbed_x = mean_x + std_x[:, None, None] * z_x
        perturbed_x = mask_x(perturbed_x, flags)

        z_adj = gen_noise(adj, flags, sym=self.sde_adj.sym) 
        mean_adj, std_adj = self.sde_adj.marginal_prob(adj, t)
        perturbed_adj = mean_adj + std_adj[:, None, None] * z_adj
        perturbed_adj = mask_adjs(perturbed_adj, flags)

        return perturbed_x, perturbed_adj, flags
        
    
    def forward(self, batch):
        x, adj = load_batch(batch, self.device) 
        perturbed_x, perturbed_adj, flags = self.perturb(x, adj)
        x, adj, _ = self.sampling_fn(self.model_x, self.model_adj, flags,
                                     x=perturbed_x, adj=perturbed_adj)
        adj = quantize(adj)
        return x, adj
        