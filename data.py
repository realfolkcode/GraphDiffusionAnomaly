import dgl
import numpy as np
from dgl.data import DGLDataset
from pygod.utils import load_data
import torch_geometric
from torch_geometric.utils import is_undirected
from torch_geometric.transforms import ToUndirected

from utils import standardize


class AnomalyDataset(DGLDataset):
    def __init__(self, name, radius=1):
        self.radius = radius
        super().__init__(name=name)
    
    def process(self):
        data = load_data(self.name)

        if not is_undirected(data['edge_index']):
            data = ToUndirected()(data)
        assert is_undirected(data['edge_index'])

        data['x'] = standardize(data['x'])

        graph = dgl.from_networkx(
                  torch_geometric.utils.to_networkx(data,
                                                    node_attrs=['x']),
                  node_attrs=['x'])

        self.feat_dim = graph.ndata['x'].shape[1]
        self._adjust_max_node_num(graph)

        # Create and store ego graphs
        self.ego_graphs = []
        for idx in range(len(graph.nodes())):
            g = dgl.khop_out_subgraph(graph, idx, self.radius)[0]
            g = self._sample_subgraph(g, self.max_node_num)
            self.ego_graphs.append(g)
    
    def _sample_subgraph(self, g, max_node_num):
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
    
    def _adjust_max_node_num(self, graph):
        ego_lens = graph.out_degrees().numpy()
        self.max_node_num = int(np.quantile(ego_lens, 0.95))

    def __getitem__(self, idx):
        return self.ego_graphs[idx]
    
    def __len__(self):
        return len(self.ego_graphs)
