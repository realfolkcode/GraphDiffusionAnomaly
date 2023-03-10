import torch
import torchode
import numpy as np
from scipy import integrate

from GDSS.losses import get_score_fn
from GDSS.utils.loader import load_device, load_seed, load_model_from_ckpt, \
                              load_ema_from_ckpt, load_ckpt, load_sampling_fn, \
                              load_batch, load_sde
from GDSS.utils.graph_utils import node_flags, mask_x, mask_adjs, \
                                   quantize, count_nodes


def get_div_fn(fn_x, fn_adj):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, adj, t, eps_x, eps_adj):
    with torch.enable_grad():
      x.requires_grad_(True)
      adj.requires_grad_(True)
      xx = fn_x(x, adj, t)
      aa = fn_adj(x, adj, t)
      fn_eps_x = torch.sum(xx * eps_x)
      grad_fn_eps_x = torch.autograd.grad(fn_eps_x, x)[0]
      fn_eps_adj = torch.sum(adj * eps_adj)
      grad_fn_eps_adj = torch.autograd.grad(fn_eps_adj, adj)[0]
      div = torch.sum(grad_fn_eps_x * eps_x, dim=[1,2]) + torch.sum(grad_fn_eps_adj * eps_adj, dim=[1,2])
    x.requires_grad_(False)
    adj.requires_grad_(False)
    
    return div

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
    return get_div_fn(lambda xx, aa, tt: drift_fn(model_x, xx, aa, flags, tt, False),
                      lambda xx, aa, tt: drift_fn(model_adj, xx, aa, flags, tt, True))(x, adj, t, noise_x, noise_adj)

  def likelihood_fn(model_x, model_adj, x, adj, flags):
    with torch.no_grad():
      shape_x = x.shape
      shape_adj = adj.shape
      bs = shape_x[0]
      epsilon_x = mask_x(torch.randn_like(x), flags)
      epsilon_adj = mask_adjs(torch.randn_like(adj), flags)
      if sde_adj.sym:
         epsilon_adj = epsilon_adj.triu(1)

      def ode_func(t, g):
        len_flat_x = shape_x[1] * shape_x[2]
        xx = g[:, :len_flat_x].reshape(shape_x)
        aa = g[:, len_flat_x:-1].reshape(shape_adj)

        vec_t = torch.ones(bs, device=x.device) * t

        drift_x = drift_fn(model_x, xx, aa, flags, vec_t, is_adj=False)
        drift_x = drift_x.reshape((bs, -1))
        drift_adj = drift_fn(model_adj, xx, aa, flags, vec_t, is_adj=True)
        drift_adj = drift_adj.reshape((bs, -1))
        drift = torch.concat((drift_x, drift_adj), -1)

        logp_grad = div_fn(model_x, model_adj, xx, aa, flags, vec_t, epsilon_x, epsilon_adj).reshape((bs, -1))
        return torch.concat((drift, logp_grad), -1)

      init = torch.concat([x.reshape((bs, -1)),
                                adj.reshape((bs, -1)), 
                                torch.zeros((bs, 1)).to(x.device)], axis=-1)
      term = torchode.ODETerm(ode_func)
      step_method = torchode.Dopri5(term=term)
      step_size_controller = torchode.IntegralController(atol=atol, rtol=rtol, term=term)
      adjoint = torchode.AutoDiffAdjoint(step_method, step_size_controller).to(x.device)

      t_eval = torch.Tensor([eps, sde_x.T]).repeat((bs,1)).to(init.device)
      problem = torchode.InitialValueProblem(y0=init, t_eval=t_eval)
      solution = adjoint.solve(problem)

      zp = solution.ys[:, -1, :]
      len_flat_x = shape_x[1] * shape_x[2]
      z_x = zp[:, :len_flat_x].reshape(x.shape)
      z_adj = zp[:, len_flat_x:-1].reshape(adj.shape)
      delta_logp = zp[:, -1]

      num_nodes = count_nodes(adj)
      N_x = num_nodes * shape_x[-1]
      N_adj = num_nodes**2 - num_nodes
      if sde_adj.sym:
          prior_logp = sde_x.prior_logp(z_x, N_x) + sde_adj.prior_logp(z_adj, N_adj) / 2
      else:
          prior_logp = sde_x.prior_logp(z_x, N_x) + sde_adj.prior_logp(z_adj, N_adj)

      nll = -(prior_logp + delta_logp)
      return nll

  return likelihood_fn


class LikelihoodEstimator(torch.nn.Module):
    def __init__(self, config, num_sample):
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
        self.num_sample = num_sample
        
    
    def forward(self, batch):
        x, adj = load_batch(batch, self.device)
        flags = node_flags(adj)
        likelihood = 0
        for i in range(self.num_sample):
            likelihood = likelihood + self.likelihood_fn(self.model_x, self.model_adj, x, adj, flags)
        likelihood /= self.num_sample
        return likelihood
