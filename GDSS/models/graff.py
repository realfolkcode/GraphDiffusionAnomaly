import torch
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as F
import math
from typing import Any


# -------- GRAFF layer --------
class DenseGRAFFConv(torch.nn.Module):
    def __init__(self, nhid):
        super(DenseGRAFFConv, self).__init__()
        self.ext_lin = External(nhid)
        self.pair_lin = Pairwise(nhid)
        self.source_lin = Source()

    def forward(self, x, adj, x0, mask=None, add_loop=True):
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
        B, N, _ = adj.size()

        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            adj[:, idx, idx] = 1

        # Linearly transform node feature matrix.
        out = self.pair_lin(x)

        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj, out)

        # Add the external and source contributions
        out -= self.ext_lin(x) + self.source_lin(x0)

        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__)


class External(torch.nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((1, num_features)))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)

    def forward(self, x):
        return x * self.weight


class PairwiseParametrization(torch.nn.Module):
    def forward(self, W):
        # Construct a symmetric matrix with zero diagonal
        W0 = W[:, :-2].triu(1)
        W0 = W0 + W0.T

        # Retrieve the `q` and `r` vectors from the last two columns
        q = W[:, -2]
        r = W[:, -1]
        # Construct the main diagonal
        w_diag = torch.diag(q * torch.sum(torch.abs(W0), 1) + r) 

        return W0 + w_diag


class Pairwise(torch.nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        # Pay attention to the dimensions
        self.lin = torch.nn.Linear(num_hidden + 2, num_hidden, bias=False)
        # Add parametrization
        parametrize.register_parametrization(self.lin, "weight", PairwiseParametrization(), unsafe=True)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin.reset_parameters()
    
    def forward(self, x):
        return self.lin(x)


class Source(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(1))
        self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight)
    
    def forward(self, x):
        return x * self.weight
