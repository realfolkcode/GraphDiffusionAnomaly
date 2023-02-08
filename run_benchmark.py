import dgl
import torch
import torch_geometric
import numpy as np
import argparse
from random import choice

from GDSS.parsers.config import get_config
from GDSS.trainer import Trainer
from GDSS.utils.data_loader import dataloader

from data import AnomalyDataset
from anomaly_scores import save_final_scores


def run_experiment(config, dataset, exp_name, **kwargs):
    # Train GDSS
    trainer = Trainer(config)
    train_loader = dataloader(config, dataset, drop_last=False)
    trainer.train_loader = train_loader
    ckpt = trainer.train(exp_name)
    config.ckpt = ckpt

    trajectory_sample = kwargs['trajectory_sample']
    num_sample = kwargs['num_sample']

    # Inference
    save_final_scores(config, dataset, exp_name, trajectory_sample, num_sample, 
                      save_intermediate=False)


def draw_hyperparameters(config, dataset_name):
    lr = [0.1, 0.05, 0.01]
    weight_decay = 0.01

    if dataset_name == 'reddit':
        # for the low feature dimension dataset
        hid_dim = [32, 48, 64]
    elif dataset_name in ['enron', 'disney', 'dgraph', 'books']:
        hid_dim = [8, 12, 16]
    else:
        hid_dim = [32, 64, 128, 256]

    config.train.lr = choice(lr)
    config.train.weight_decay = weight_decay
    config.model.hdim = choice(hid_dim)
    config.model.adim = config.model.hdim

    return config


def run_benchmark(args):
    num_trials = 20

    config = get_config(args.config, args.seed)
    exp_name = args.exp_name

    # Load dataset
    dataset_name = config.data.data
    dataset = AnomalyDataset(dataset_name, radius=1)
    print(f'Dataset: {dataset_name}')
    print(f'Number of nodes: {len(dataset)}')

    # Adjust max node num
    config.data.max_node_num = dataset.max_node_num
    print(f'Max size subgraphs (95% quantile): {config.data.max_node_num}')

    # Adjust feature dimension
    config.data.max_feat_num = dataset.feat_dim
    print(f'Feature dimension: {config.data.max_feat_num}')

    config.train.print_interval = 50

    for i in range(num_trials):
        print(f'Running experiment no. {i}')
        config = draw_hyperparameters(config, dataset_name)
        run_experiment(config, dataset, f'{exp_name}_{i}', 
                       trajectory_sample=args.trajectory_sample, num_sample=args.num_sample)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config path')
    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--trajectory_sample', type=int, default=1, required=False, help='number of samples per trajectory')
    parser.add_argument('--num_sample', type=int, default=1, required=False, help='number of samples per node')
    parser.add_argument('--seed', type=int, default=42, required=False, help='rng seed value')
    args = parser.parse_args()
    run_benchmark(args)