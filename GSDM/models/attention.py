import math
import torch
from torch.nn import Parameter
import torch.nn.functional as F

from .layers import DenseGCNConv, MLP
from ..utils.graph_utils import mask_adjs, mask_x


# -------- Graph Multi-Head Attention (GMH) --------
# -------- From Baek et al. (2021) --------
class Attention(torch.nn.Module):

    def __init__(self, in_dim, cond_dim, attn_dim, out_dim, num_heads=4, conv='GCN'):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.attn_dim = attn_dim
        self.out_dim = out_dim
        self.conv = conv

        self.gnn_q, self.gnn_k, self.gnn_v = self.get_gnn(in_dim, cond_dim, attn_dim, out_dim, conv)
        self.activation = torch.tanh 
        self.softmax_dim = 2

    def forward(self, x, cond, flags, attention_mask=None):

        Q = self.gnn_q(x) 
        K = self.gnn_k(cond)
        V = self.gnn_v(cond)
        
        dim_split = self.attn_dim // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)

        if attention_mask is not None:
            attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
            attention_score = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.out_dim)
            A = self.activation( attention_mask + attention_score )
        else:
            A = self.activation( Q_.bmm(K_.transpose(1,2))/math.sqrt(self.out_dim) ) # (B x num_heads) x N x N
        
        # -------- (B x num_heads) x N x N --------
        A = A.view(-1, x.shape[0], x.shape[1], x.shape[1])
        A = A.mean(dim=0)
        V = torch.matmul(A, V)

        return V

    def get_gnn(self, in_dim, cond_dim, attn_dim, out_dim, conv='GCN'):

        if conv == 'vanilla':
            gnn_q = torch.nn.Linear(in_dim, attn_dim)
            gnn_k = torch.nn.Linear(cond_dim, attn_dim)
            gnn_v = torch.nn.Linear(cond_dim, out_dim)
            return gnn_q, gnn_k, gnn_v

        else:
            raise NotImplementedError(f'{conv} not implemented.')


# -------- Layer of ScoreNetworkA --------
class AttentionLayer(torch.nn.Module):

    def __init__(self, num_linears, conv_input_dim, cond_dim, attn_dim, conv_output_dim, input_dim, output_dim, 
                    num_heads=4, conv='GCN'):

        super(AttentionLayer, self).__init__()

        self.attn = torch.nn.ModuleList()
        for _ in range(input_dim):
            self.attn_dim =  attn_dim
            # Self-attention
            self.attn.append(Attention(conv_input_dim, conv_input_dim, self.attn_dim, conv_output_dim,
                                       num_heads=num_heads, conv=conv))
            # Cross-attention
            self.attn.append(Attention(conv_input_dim, cond_dim, self.attn_dim, conv_output_dim,
                                        num_heads=num_heads, conv=conv))
        input_dim = input_dim * 2
        self.hidden_dim = 2*max(input_dim, output_dim)
        self.multi_channel = MLP(2, input_dim*conv_output_dim, self.hidden_dim, conv_output_dim, 
                                    use_bn=False, activate_func=F.elu)

    def forward(self, x, cond, flags):
        """

        :param x:  B x N x F_i
        :param cond: B x N x F_cond
        :return: x_out: B x N x F_o
        """
        x_list = []
        for i in range(len(self.attn)):
            if i % 2 == 0:
                # Self-attention
                _x = self.attn[_](x, x, flags)
            else:
                # Cross-attention
                _x = self.attn[_](x, cond, flags)
            x_list.append(_x)
        x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)), flags)
        x_out = torch.tanh(x_out)

        return x_out
