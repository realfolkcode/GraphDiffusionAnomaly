# Anomaly Detection in Networks via Score-Based Generative Models
This is a repository for an [ICML 2023 SPIGM Workshop paper](https://arxiv.org/abs/2306.15324) and my Master's Thesis at Skoltech.

**Authors:** Dmitrii Gavrilev, Evgeny Burnaev (research advisor)

In this project, we use [GDSS](https://github.com/harryjo97/GDSS) as a generative model.

## Abstract
Node outlier detection in attributed graphs is a challenging problem for which there is no method that would work well across different datasets. Motivated by the state-of-the-art results of score-based models in graph generative modeling, we propose to incorporate them into the aforementioned problem. Our method achieves competitive results on small-scale graphs. We provide an empirical analysis of the Dirichlet energy, and show that generative models might struggle to accurately reconstruct it.

## Prerequisites
- Install [DGL](https://www.dgl.ai/pages/start.html) and [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
- `pip install -r requirements.txt`

## Usage
`python run_benchmark.py` trains GDSS with random hyperparameters on a chosen dataset, runs inference with our methods, and repeats this pipeline 20 times
Arguments:
- `--config` (path to a dataset config)
- `--exp_name` (the name of the experiment/checkpoints)
- `--radius` (the number of hops in ego-graphs)
- `--trajectory_sample` (the number of samples per trajectory; $K$ in the paper)
- `--num_sample` (the number of samples per node; $S$ in the paper)
- `--num_steps` (the number of steps to denoise for the full time horizon $[0,1]$)
- `--skip_training` (inference-only mode; assumes the checkpoints already exist)
