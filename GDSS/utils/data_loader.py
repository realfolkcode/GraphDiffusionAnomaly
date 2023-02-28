import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import dgl
from dgl.dataloading import GraphDataLoader
import numpy as np
from functools import partial

from .graph_utils import pad_adjs


def collate_fn(graphs, max_node_num, dequantize):
    graph_list = [g.adj().to_dense() for g in graphs]

    graph_list = [pad_adjs(g, max_node_num) for g in graph_list]
    adjs_tensor = torch.stack(graph_list)

    x_tensor = torch.stack([F.pad(g.ndata['x'], 
                                  (0, 0, 0, max(max_node_num - len(g.nodes()), 0)), 
                                  "constant", 0)
                            for g in graphs])

    if dequantize:
        noise = torch.rand_like(adjs_tensor) / 2
        adjs_tensor -= torch.sign(adjs_tensor - 0.5) * noise

    return x_tensor, adjs_tensor


def dataloader(config, dataset, shuffle=True, drop_last=True, dequantize=False):
    loader = GraphDataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config.data.batch_size,
        drop_last=drop_last,
        collate_fn=partial(collate_fn, max_node_num=config.data.max_node_num, dequantize=dequantize)
    )
    return loader
