import sys
import os
import random
import torch
import numpy as np
from nosalro.vae import VariationalAutoencoder, StatesDataset, train, visualize, Scaler, Shuffle, AngleToSinCos, Transform
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    shuffle = Shuffle(seed=42)
    angle_to_sin_cos = AngleToSinCos(angle_column=2)
    scaler = Scaler()
    target_scaler = Scaler()

    transforms = Transform([
        shuffle,
        angle_to_sin_cos,
        scaler.fit,
        scaler
        ])

    target_transforms = Transform([
        shuffle,
        # angle_to_sin_cos,
        target_scaler.fit,
        target_scaler
        ])

    dataset = StatesDataset(path='data/no_wall.dat')
    target_dataset = StatesDataset(path='data/no_wall.dat')

    dataset.set_data(transforms(dataset[:]))
    target_dataset.set_data(target_transforms(target_dataset[:]))

    vae = VariationalAutoencoder(4, 2, output_dims=3, hidden_sizes=[32,32], scaler=scaler).to(device)
    epochs = 500
    lr = 3e-04
    vae = train(
        vae,
        epochs,
        lr,
        dataset,
        device,
        beta = 10,
        file_name = 'models/vae_models/no_wall_vae.pt',
        overwrite = False,
        weight_decay = 0,
        batch_size = 1024,
        target_dataset=target_dataset
    )

    i, x_hat_var, mu, logvar = vae(torch.tensor(dataset[:]).to(device), device, False, scale=False)
    visualize([dataset[:], i.detach().cpu().numpy()], projection='2d')
