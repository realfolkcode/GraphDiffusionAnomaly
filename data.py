import dgl
import numpy as np
from dgl.data import DGLDataset
from pygod.utils import load_data
import torch_geometric
import torch
import torch.nn.functional as F
from torch_geometric.utils import is_undirected
from torch_geometric.transforms import ToUndirected
from tqdm import tqdm

from utils import standardize


class AnomalyDataset(DGLDataset):
    def __init__(self, name, num_partition, radius=1, undirected=True):
        self.num_partition = num_partition
        self.radius = radius
        self.undirected = undirected
        super().__init__(name=name)
    
    def process(self):
        data = load_data(self.name)

        if self.undirected:
            data = ToUndirected()(data)

        data['x'] = standardize(data['x'])

        graph = dgl.from_networkx(
                  torch_geometric.utils.to_networkx(data,
                                                    node_attrs=['x']),
                  node_attrs=['x'])
        
        self._create_pe(graph)

        self.feat_dim = graph.ndata['x'].shape[1]
        self._adjust_max_node_num(graph)

        # Create and store ego graphs
        self.ego_graphs = []
        for idx in tqdm(range(len(graph.nodes()))):
            g_out, _ = dgl.khop_out_subgraph(graph, idx, self.radius)
            g_in, _ = dgl.khop_in_subgraph(graph, idx, self.radius)
            ego_indices = torch.concat((g_out.ndata[dgl.NID], g_in.ndata[dgl.NID])).unique()

            g = dgl.node_subgraph(graph, ego_indices)
            center_idx = torch.argwhere(g.ndata[dgl.NID] == idx).item()

            g = self._sample_subgraph(g, center_idx, self.max_node_num)
            self.ego_graphs.append(g)
    
    def _create_pe(self, graph):
        pe = torch.zeros(graph.num_nodes()).long()
        partition = dgl.metis_partition(graph, k=self.num_partition, reshuffle=False)
        for part_idx in partition:
            part_graph = partition[part_idx]
            pe[part_graph.ndata['_ID']] = part_idx
        pe = F.one_hot(pe)
        graph.ndata['pe'] = pe
    
    def _sample_subgraph(self, g, center_idx, max_node_num):
        if len(g.nodes()) <= max_node_num:
            return g
        g_undirected = dgl.to_bidirected(g)
        idx = []
        for bfs_nodes in dgl.bfs_nodes_generator(g_undirected, center_idx):
            remaining = max_node_num - len(idx)
            if remaining <= 0:
                break
            k = min(remaining, len(bfs_nodes))
            idx += bfs_nodes[:k]
        idx = idx[:max_node_num]
        return dgl.node_subgraph(g, idx)
    
    def _adjust_max_node_num(self, graph):
        ego_lens = graph.out_degrees().numpy()
        self.max_node_num = int(np.quantile(ego_lens, 0.95))
        self.max_node_num = min(50, self.max_node_num)

    def __getitem__(self, idx):
        return self.ego_graphs[idx]
    
    def __len__(self):
        return len(self.ego_graphs)
