import torch
import numpy as np
from tqdm import tqdm

from GDSS.utils.data_loader import dataloader
from GDSS.utils.graph_utils import adjs_to_graphs, count_nodes
from GDSS.utils.plot import plot_graphs_list
from GDSS.reconstruction import Reconstructor


def calculate_scores(config, loader, data_len, exp_name, num_sample=1, plot_graphs=True, save_res=True):
    reconstructor = Reconstructor(config)

    x_scores = torch.zeros(data_len)
    adj_scores = torch.zeros(data_len)

    gen_graph_list = []
    orig_graph_list = []

    batch_start_pos = 0
    for i, batch in tqdm(enumerate(loader)):
        x = batch[0]
        adj = batch[1]

        # Normalization terms (number of nodes in each graph and number of features)
        num_nodes = count_nodes(adj)
        num_feat = x.shape[2]

        x_err = torch.zeros(x.shape[0])
        adj_err = torch.zeros(adj.shape[0])

        for _ in range(num_sample):
            with torch.no_grad():
                x_reconstructed, adj_reconstructed = reconstructor(batch)
            x_reconstructed = x_reconstructed.to('cpu')
            adj_reconstructed = adj_reconstructed.to('cpu')
            
            x_err = x_err + torch.linalg.norm(x - x_reconstructed, dim=[1, 2]) / (num_nodes * num_feat)
            adj_err = adj_err + torch.linalg.norm(adj - adj_reconstructed, dim=[1, 2]) / (num_nodes**2)

        bs = x.shape[0]
        batch_end_pos = batch_start_pos + bs

        x_scores[batch_start_pos:batch_end_pos] = x_err
        adj_scores[batch_start_pos:batch_end_pos] = adj_err

        batch_start_pos = batch_end_pos

        # Convert the first batch to networkx for plotting
        if i == 0 and plot_graphs:
            eps = 1e-9
            rel_x_err = torch.linalg.norm(x - x_reconstructed, dim=[2])
            rel_x_err /= (torch.linalg.norm(x, dim=[2]) + eps)

            nx_graphs, empty_nodes = adjs_to_graphs(adj.numpy(), False, return_empty=True)
            orig_graph_list.extend(nx_graphs)
            gen_graph_list.extend(adjs_to_graphs(adj_reconstructed.numpy(), False, empty_nodes=empty_nodes))
    
    if plot_graphs:
        pos_list = plot_graphs_list(graphs=orig_graph_list, title=f'orig_{exp_name}', max_num=16, save_dir='./')
        _ = plot_graphs_list(graphs=gen_graph_list, title=f'reconstruction_{exp_name}', max_num=16, save_dir='./', 
                            pos_list=pos_list, rel_x_err=rel_x_err)
    
    if save_res:
        with open(f'{exp_name}_scores.npy', 'wb') as f:
            np.save(f, x_scores.numpy())
            np.save(f, adj_scores.numpy())
    
    return x_scores, adj_scores


def save_final_scores(config, dataset, exp_name, trajectory_sample, num_sample=1, num_steps=100, save_intermediate=True):
    loader = dataloader(config, 
                        dataset,
                        shuffle=False,
                        drop_last=False)
    data_len = len(dataset)
    
    endtime = config.sde.adj.endtime
    T_lst = np.linspace(0, endtime, trajectory_sample + 2, endpoint=True)[1:-1]
    default_num_scales = config.sde.adj.num_scales

    x_scores_final = 0
    adj_scores_final = 0

    for T in T_lst:
        config.sde.x.endtime = T
        config.sde.adj.endtime = T
        new_num_scales = int(T * default_num_scales)
        config.sde.x.num_scales = new_num_scales
        config.sde.adj.num_scales = new_num_scales

        new_exp_name = f'{exp_name}_scales_{new_num_scales}'
        x_scores, adj_scores = calculate_scores(config, loader, data_len, new_exp_name,
                                                num_sample=num_sample, plot_graphs=False,
                                                save_res=save_intermediate)
        x_scores_final = x_scores_final + x_scores
        adj_scores_final = adj_scores_final + adj_scores
    
    with open(f'{exp_name}_final_scores.npy', 'wb') as f:
        np.save(f, x_scores_final.numpy())
        np.save(f, adj_scores_final.numpy())
