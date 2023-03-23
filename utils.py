import torch
import numpy as np


def standardize(x, eps=1e-7):
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x = (x - m) / (s + eps)
    return x


def calculate_snr(T_lst, config_sde, eps=1e-7):
    sde_type = config_sde.type
    if sde_type == 'VP':
        beta_min = config_sde.beta_min
        beta_max = config_sde.beta_max
        integral = -beta_min * T_lst - (beta_max - beta_min) / 2 * T_lst**2
        mu_squared = np.exp(integral)
        sigma_squared = 1 - np.exp(integral)
        return mu_squared / (sigma_squared + eps)
    else:
        raise NotImplementedError(f'{sde_type} not implemented.')
