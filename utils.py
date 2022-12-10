import numpy as np
import dgl


def sample_subgraph(g, max_node_num):
    if len(g.nodes()) <= max_node_num:
        return g
    idx = []
    idx += np.random.choice(np.arange(1, len(g.nodes())), 
                            max_node_num,
                            replace=False).tolist()
    return dgl.node_subgraph(g, idx)