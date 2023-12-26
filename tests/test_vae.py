import sys
import os
import random
import torch
import numpy as np
from agrl.vae import VariationalAutoencoder, StatesDataset, train, visualize
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import matplotlib.pyplot as plt

def random_points_generator(row, columns, dist, **kwargs):
    points = []
    for i in range(row):
        tmp_points = []
        for j in range(columns):
            tmp_points.append(dist(**kwargs))
        points.append(tmp_points)
    return np.array(points, dtype=np.float32)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shuffle = Shuffle(seed=42)
    angle_to_sin_cos = AngleToSinCos(angle_column=2)
    scaler = Scaler()
    target_scaler = Scaler()

    transforms = Compose([
        shuffle,
        angle_to_sin_cos,
        scaler.fit,
        scaler
        ])

    dataset = StatesDataset(path='data/go_explore_600.dat', transforms=transforms)

    vae = VariationalAutoencoder(4, 3, output_dims=4, hidden_sizes=[32,32], scaler=scaler).to(device)
    epochs = 2000
    lr = 1e-04
    vae = train(
        vae,
        epochs,
        lr,
        dataset,
        device,
        beta = 0,
        file_name = 'models/vae_models/no_wall_vae.pt',
        overwrite = True,
        weight_decay = 0,
        batch_size = 1024,
    )

    i, x_hat_var, mu, logvar = vae(torch.tensor(dataset[:]).to(device), device, False, False)
    visualize([dataset[:], i.detach().cpu().numpy()], projection='2d')