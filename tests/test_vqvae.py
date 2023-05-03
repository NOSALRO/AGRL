import os
import copy
from nosalro.vae import StatesDataset
from nosalro.transforms import Compose, AngleToSinCos, Scaler, Shuffle
from nosalro.vae import VQVAE
from nosalro.vae import visualize
import numpy as np
import torch
import matplotlib.pyplot as plt



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = Scaler()
    shuffler = Shuffle()
    transforms = Compose([shuffler, scaler.fit, scaler])
    dataset = StatesDataset(path='data/go_explore_xy.dat', transforms=transforms)

    vqvae = VQVAE(input_dims=2, latent_dims=8, codes=10, output_dims=2, beta=0.25, hidden_sizes = [64, 64])
    vqvae = vqvae.to(device)

    vqvae = vqvae.train(
        loss_fn=vqvae.loss_fn,
        model=vqvae,
        epochs=700,
        lr=3e-4,
        dataset = dataset,
        device = device,
        overwrite=False,
        file_name = '.tmp/vqvae_test',
        batch_size = 256,
        reconstruction_type='mse'
    )

    x_hat, x_hat_var, one_hot, _ = vqvae(torch.tensor(dataset[:], device=device), device, False, False)
    print(one_hot)
    x_hat = x_hat.detach().cpu().numpy()
    visualize([scaler(dataset[:], True), scaler(x_hat[:], True)], file_name='.tmp/plot')
