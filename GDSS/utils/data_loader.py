import torch
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
import dgl
from dgl.dataloading import GraphDataLoader
import numpy as np
from functools import partial

from .graph_utils import pad_adjs


def sample_subgraph(g, max_node_num):
    if len(g.nodes()) <= max_node_num:
        return g
    idx = []
    for bfs_nodes in dgl.bfs_nodes_generator(g, 0):
        remaining = max_node_num - len(idx)
        if remaining <= 0:
            break
        k = min(remaining, len(bfs_nodes))
        idx += bfs_nodes[:k]
    idx = idx[:max_node_num]
    return dgl.node_subgraph(g, idx)


def collate_fn(graphs, max_node_num):
    graphs = [sample_subgraph(g, max_node_num) for g in graphs]

    graph_list = [g.adj().to_dense() for g in graphs]

    graph_list = [pad_adjs(g, max_node_num) for g in graph_list]
    adjs_tensor = torch.stack(graph_list)

    x_tensor = torch.stack([F.pad(g.ndata['x'], 
                                  (0, 0, 0, max(max_node_num - len(g.nodes()), 0)), 
                                  "constant", 0)
                            for g in graphs])

    return x_tensor, adjs_tensor


def dataloader(config, dataset, shuffle=True, drop_last=True):
    loader = GraphDataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=config.data.batch_size,
        drop_last=drop_last,
        collate_fn=partial(collate_fn, max_node_num=config.data.max_node_num)
    )
    return loader
