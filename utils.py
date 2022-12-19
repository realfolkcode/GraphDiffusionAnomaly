import torch


def standardize(x, eps=1e-7):
    m = x.mean(0, keepdim=True)
    s = x.std(0, unbiased=False, keepdim=True)
    x = (x - m) / (s + eps)
    return x