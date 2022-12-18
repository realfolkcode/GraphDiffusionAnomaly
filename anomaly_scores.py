import torch
from tqdm import tqdm

from GDSS.utils.data_loader import dataloader
from GDSS.reconstruction import Reconstructor


def calculate_scores(config, dataset):
    reconstructor = Reconstructor(config)
    loader = dataloader(config, 
                        dataset,
                        shuffle=False,
                        drop_last=False)

    x_scores = torch.zeros(len(dataset))
    adj_scores = torch.zeros(len(dataset))

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
    
    return x_scores, adj_scores