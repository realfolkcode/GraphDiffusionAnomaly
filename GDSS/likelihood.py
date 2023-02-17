import torch
import numpy as np
from scipy import integrate

from GDSS.losses import get_score_fn
from GDSS.utils.loader import load_device, load_seed, load_model_from_ckpt, \
                              load_ema_from_ckpt, load_ckpt, load_sampling_fn, \
                              load_batch, load_sde
from GDSS.utils.graph_utils import node_flags, gen_noise, mask_x, mask_adjs, \
                                   quantize


def get_div_fn(fn_x, fn_adj):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, adj, t, eps_x, eps_adj):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps_x = torch.sum(fn_x(x, t) * eps_x)
      grad_fn_eps_x = torch.autograd.grad(fn_eps_x, x)[0]
    x.requires_grad_(False)

    with torch.enable_grad():
      adj.requires_grad_(True)
      fn_eps_adj = torch.sum(fn_adj(adj, t) * eps_adj)
      grad_fn_eps_adj = torch.autograd.grad(fn_eps_adj, adj)[0]
    adj.requires_grad_(False)
    
    return torch.sum(grad_fn_eps_x * eps_x, dim=tuple(range(1, len(x.shape)))) + \
           torch.sum(grad_fn_eps_adj * eps_adj, dim=tuple(range(1, len(adj.shape))))

  return div_fn


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_likelihood_fn(sde_x, sde_adj,
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
  def drift_fn(model, x, adj, flags, t, is_adj):
    """Get the drift function of the reverse-time SDE."""
    if is_adj:
      score_fn = get_score_fn(sde_adj, model, train=False, continuous=True)
      rsde = sde_adj.reverse(score_fn, probability_flow=True)
      return rsde.sde(x, adj, flags, t, is_adj=True)[0]
    else:
      score_fn = get_score_fn(sde_x, model, train=False, continuous=True)
      rsde = sde_x.reverse(score_fn, probability_flow=True)
      return rsde.sde(x, adj, flags, t, is_adj=False)[0]

  def div_fn(model_x, model_adj, x, adj, flags, t, noise_x, noise_adj):
    return get_div_fn(lambda xx, tt: drift_fn(model_x, xx, adj, flags, tt, False),
                      lambda aa, tt: drift_fn(model_adj, x, aa, flags, tt, True))(x, adj, t, noise_x, noise_adj)

  def likelihood_fn(model_x, model_adj, x, adj, flags):
    with torch.no_grad():
      shape_x = x.shape
      shape_adj = adj.shape
      epsilon_x = torch.randn_like(x)
      epsilon_adj = torch.randn_like(adj)
      bs = shape_x[0]

      def ode_func(t, g):
        len_flat_x = shape_x[0] * shape_x[1] * shape_x[2]
        xx = g[:len_flat_x]
        aa = g[len_flat_x:-bs]
        xx = from_flattened_numpy(xx, shape_x).to(x.device).type(torch.float32)
        aa = from_flattened_numpy(aa, shape_adj).to(adj.device).type(torch.float32)

        vec_t = torch.ones(bs, device=x.device) * t

        drift_x = drift_fn(model_x, xx, aa, flags, vec_t, is_adj=False)
        drift_x = to_flattened_numpy(drift_x)
        drift_adj = drift_fn(model_adj, xx, aa, flags, vec_t, is_adj=True)
        drift_adj = to_flattened_numpy(drift_adj)
        drift = np.concatenate((drift_x, drift_adj))

        logp_grad = to_flattened_numpy(div_fn(model_x, model_adj, xx, aa, flags, vec_t, epsilon_x, epsilon_adj))
        return np.concatenate([drift, logp_grad], axis=0)

      init = np.concatenate([to_flattened_numpy(x),
                             to_flattened_numpy(adj), 
                             np.zeros((bs,))], axis=0)
      solution = integrate.solve_ivp(ode_func, (eps, sde_x.T), init, rtol=rtol, atol=atol, method=method)
      zp = solution.y[:, -1]
      len_flat_x = shape_x[0] * shape_x[1] * shape_x[2]
      z_x = from_flattened_numpy(zp[:len_flat_x], shape_x).to(x.device).type(torch.float32)
      z_adj = from_flattened_numpy(zp[len_flat_x:-bs], shape_adj).to(adj.device).type(torch.float32)
      delta_logp = from_flattened_numpy(zp[-bs:], (bs,)).to(x.device).type(torch.float32)
      prior_logp = sde_x.prior_logp(z_x) + sde_adj.prior_logp(z_adj)
      nll = -(prior_logp + delta_logp)
      return nll

  return likelihood_fn


class LikelihoodEstimator(torch.nn.Module):
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

        self.sde_x = load_sde(config.sde.x)
        self.sde_adj = load_sde(config.sde.adj)

        self.likelihood_fn = get_likelihood_fn(self.sde_x, self.sde_adj)
        
    
    def forward(self, batch):
        x, adj = load_batch(batch, self.device)
        flags = node_flags(adj)
        likelihood = self.likelihood_fn(self.model_x, self.model_adj, x, adj, flags)
        return likelihood
        