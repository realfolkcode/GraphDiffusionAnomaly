# Anomaly Detection in Networks via Score-Based Models
This is a repository for my Master's Thesis at Skoltech

In this project, we use [GDSS](https://github.com/harryjo97/GDSS) as a generative model.

### Abstract

Node outlier detection in attributed graphs is a challenging problem for which there is no method that would work well across different datasets. Motivated by the state-of-the-art results of score-based models in graph generative modeling, we propose to incorporate them into the aforementioned problem. We leverage score-based models by first constructing ego-centric graphs for each node. Then, we assign anomaly scores based on the dissimilarity between ego-graphs and their reconstructions or density estimation. Our method achieves competitive results on small-scale graphs. We provide an empirical analysis of the Dirichlet energy and show that generative models might struggle to accurately reconstruct it.
