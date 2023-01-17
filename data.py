import dgl
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
        
        self.graph = graph
        self.feat_dim = self.graph.ndata['x'].shape[1]
    
    def __getitem__(self, idx):
        ego_graph = dgl.khop_out_subgraph(self.graph, idx, self.radius)[0]
        return ego_graph
    
    def __len__(self):
        return len(self.graph.nodes())