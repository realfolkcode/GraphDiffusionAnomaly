import torch
import numpy as np
from tqdm import tqdm

from GSDM.utils.data_loader import dataloader
from GSDM.utils.graph_utils import adjs_to_graphs
from GSDM.utils.plot import plot_graphs_list
from GSDM.reconstruction import Reconstructor


def calculate_scores(config, dataset, exp_name):
    reconstructor = Reconstructor(config)
    loader = dataloader(config, 
                        dataset,
                        shuffle=False,
                        drop_last=False)

    x_scores = torch.zeros(len(dataset))
    adj_scores = torch.zeros(len(dataset))

    gen_graph_list = []
    orig_graph_list = []

    for i, batch in tqdm(enumerate(loader)):
        x = batch[0]
        adj = batch[1]

        with torch.no_grad():
            x_reconstructed, adj_reconstructed = reconstructor(batch)
        x_reconstructed = x_reconstructed.to('cpu')
        adj_reconstructed = adj_reconstructed.to('cpu')

        x_err = torch.linalg.norm(x - x_reconstructed, dim=[1, 2])
        x_err = x_err / (x.shape[1] * x.shape[2])

        adj_err = torch.linalg.norm(adj - adj_reconstructed, dim=[1, 2])
        adj_err = adj_err / (adj.shape[1] * adj.shape[2])

        bs = x.shape[0]
        x_scores[i * bs:(i+1) * bs] = x_err
        adj_scores[i * bs:(i+1) * bs] = adj_err

        # Convert the first batch to networkx for plotting
        if i == 0:
            eps = 1e-9
            rel_x_err = torch.linalg.norm(x - x_reconstructed, dim=[2])
            rel_x_err /= (torch.linalg.norm(x, dim=[2]) + eps)

            nx_graphs, empty_nodes = adjs_to_graphs(adj.numpy(), False, return_empty=True)
            orig_graph_list.extend(nx_graphs)
            gen_graph_list.extend(adjs_to_graphs(adj_reconstructed.numpy(), False, empty_nodes=empty_nodes))
    
    pos_list = plot_graphs_list(graphs=orig_graph_list, title=f'orig_{exp_name}', max_num=16, save_dir='./')
    _ = plot_graphs_list(graphs=gen_graph_list, title=f'reconstruction_{exp_name}', max_num=16, save_dir='./', 
                         pos_list=pos_list, rel_x_err=rel_x_err)
    
    with open(f'{exp_name}_scores.npy', 'wb') as f:
        np.save(f, x_scores.numpy())
        np.save(f, adj_scores.numpy())
