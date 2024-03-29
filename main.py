import dgl
import torch
import torch_geometric
import numpy as np
import argparse

from GDSS.parsers.config import get_config
from GDSS.trainer import Trainer
from GDSS.utils.data_loader import dataloader

from data import AnomalyDataset
from anomaly_scores import save_final_scores


def main(args):
    config = get_config(args.config, args.seed)
    exp_name = args.exp_name
    trajectory_sample = args.trajectory_sample
    num_sample = args.num_sample

    # Load dataset
    dataset_name = config.data.data
    dataset = AnomalyDataset(dataset_name, radius=1, undirected=config.model.sym)
    print(f'Dataset: {dataset_name}')
    print(f'Number of nodes: {len(dataset)}')

    # Adjust max node num
    config.data.max_node_num = dataset.max_node_num
    print(f'Max size subgraphs (95% quantile): {config.data.max_node_num}')

    # Adjust feature dimension
    config.data.max_feat_num = dataset.feat_dim
    print(f'Feature dimension: {config.data.max_feat_num}')

    # Train GDSS
    trainer = Trainer(config)
    train_loader = dataloader(config, dataset)
    trainer.train_loader = train_loader
    ckpt = trainer.train(exp_name)
    config.ckpt = ckpt

    # Inference
    save_final_scores(config, dataset, exp_name, trajectory_sample, num_sample)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config path')
    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--trajectory_sample', type=int, default=1, required=False, help='number of samples per trajectory')
    parser.add_argument('--num_sample', type=int, default=1, required=False, help='number of samples per node')
    parser.add_argument('--seed', type=int, default=42, required=False, help='rng seed value')
    args = parser.parse_args()
    main(args)
