import torch
import numpy as np
from tqdm import tqdm

from GDSS.utils.data_loader import dataloader
from GDSS.utils.graph_utils import adjs_to_graphs, count_nodes
from GDSS.utils.plot import plot_graphs_list
from GDSS.reconstruction import Reconstructor
from GDSS.likelihood import LikelihoodEstimator


def calculate_scores(config, loader, data_len, exp_name, num_sample=1, plot_graphs=True):
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
        
        x_err /= num_sample
        adj_err /= num_sample

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
    
    return x_scores, adj_scores


def save_final_scores(config, dataset, exp_name, trajectory_sample, num_sample=1, num_steps=100):
    loader = dataloader(config, 
                        dataset,
                        shuffle=False,
                        drop_last=False)
    data_len = len(dataset)
    
    endtime = config.sde.adj.endtime
    T_lst = np.linspace(0, endtime, trajectory_sample + 2, endpoint=True)[1:-1]

    x_scores_final = torch.zeros((data_len, trajectory_sample))
    adj_scores_final = torch.zeros((data_len, trajectory_sample))

    for i, T in enumerate(T_lst):
        config.sde.x.endtime = T
        config.sde.adj.endtime = T
        new_num_scales = int(T * num_steps)
        config.sde.x.num_scales = new_num_scales
        config.sde.adj.num_scales = new_num_scales

        new_exp_name = f'{exp_name}_scales_{new_num_scales}'
        x_scores, adj_scores = calculate_scores(config, loader, data_len, new_exp_name,
                                                num_sample=num_sample, plot_graphs=False)
        x_scores_final[:, i] = x_scores
        adj_scores_final[:, i] = adj_scores
    
    with open(f'{exp_name}_final_scores.npy', 'wb') as f:
        np.save(f, x_scores_final.numpy())
        np.save(f, adj_scores_final.numpy())


def save_likelihood_scores(config, dataset, exp_name, num_sample):
    likelihood_estimator = LikelihoodEstimator(config, num_sample)
    loader = dataloader(config, 
                        dataset,
                        shuffle=False,
                        drop_last=False,
                        dequantize=True)
    data_len = len(dataset)

    prior_constant_x = torch.zeros(data_len)
    prior_constant_adj = torch.zeros(data_len)
    prior_logp_x = torch.zeros(data_len)
    prior_logp_adj = torch.zeros(data_len)
    delta_logp = torch.zeros(data_len)
    
    batch_start_pos = 0
    for i, batch in tqdm(enumerate(loader)):
        x = batch[0]
        adj = batch[1]

        bs = x.shape[0]
        batch_end_pos = batch_start_pos + bs

        likelihood_components = likelihood_estimator(batch)
        prior_constant_x[batch_start_pos:batch_end_pos] = likelihood_components[0]
        prior_constant_adj[batch_start_pos:batch_end_pos] = likelihood_components[1]
        prior_logp_x[batch_start_pos:batch_end_pos] = likelihood_components[2]
        prior_logp_adj[batch_start_pos:batch_end_pos] = likelihood_components[3]
        delta_logp[batch_start_pos:batch_end_pos] = likelihood_components[4]

        batch_start_pos = batch_end_pos

    with open(f'{exp_name}_final_scores.npy', 'wb') as f:
        np.save(f, prior_constant_x.numpy())
        np.save(f, prior_constant_adj.numpy())
        np.save(f, prior_logp_x.numpy())
        np.save(f, prior_logp_adj.numpy())
        np.save(f, delta_logp.numpy())
