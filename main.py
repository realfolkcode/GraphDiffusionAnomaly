import dgl
import torch
import torch_geometric
from pygod.utils import load_data
import numpy as np
import networkx as nx
import time
import argparse

from GDSS.sampler import Sampler
from GDSS.parsers.config import get_config
from GDSS.trainer import Trainer
from GDSS.utils.data_loader import dataloader
from GDSS.reconstruction import Reconstructor

from data import AnomalyDataset


def main(args):
    config = get_config(args.config, args.seed)

    # Load dataset
    dataset_name = config.data.data
    dataset = AnomalyDataset(dataset_name, radius=1)
    print(f'Dataset: {dataset_name}')
    print(f'Number of nodes: {len(dataset)}')

    # Adjust max node num
    ego_lens = dataset.graph.out_degrees().numpy()
    max_node_num = int(np.quantile(ego_lens, 0.95))
    config.data.max_node_num = max_node_num
    print(f'Max size subgraphs (95% quantile): {max_node_num}')

    # Adjust feature dimension
    config.data.max_feat_num = dataset.feat_dim
    print(f'Feature dimension: {config.data.max_feat_num}')

    # Train GDSS
    trainer = Trainer(config)
    train_loader = dataloader(config, dataset)
    trainer.train_loader = train_loader
    ckpt = trainer.train(args.exp_name)
    config.ckpt = ckpt


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='config path')
    parser.add_argument('--exp_name', type=str, required=True, help='experiment name')
    parser.add_argument('--seed', type=int, default=42, required=False, help='rng seed value')
    args = parser.parse_args()
    main(args)