import sys
import os
import random
import torch
import numpy as np
from nosalro.vae import CategoricalVAE, StatesDataset, train, visualize
from nosalro.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shuffle = Shuffle(seed=42)
    scaler = Scaler()
    transforms = Compose([shuffle, scaler.fit, scaler])

    # dataset = StatesDataset(path='data/alley_right_go_explore.dat', transforms=transforms)
    x = np.linspace(0, 600, 50)
    y = np.linspace(0, 600, 50)
    grid = []
    for i in x:
        for j in y:
            grid.append([i,j])
    grid = np.array(grid)
    dataset = StatesDataset(grid, transforms=transforms)

    cvae = CategoricalVAE(2, 2, output_dims=2, hidden_sizes=[128,128], scaler=scaler).to(device)
    epochs = 200
    lr = 3e-04
    vae = cvae.train(
        cvae,
        epochs,
        lr,
        dataset,
        device,
        beta = 1,
        file_name = '.tmp/categorical_vae.pt',
        overwrite = True,
        weight_decay = 0,
        batch_size = 256,
    )

    x_hat, x_hat_var, l = vae(torch.tensor(dataset[:]).to(device), device, True, False)
    visualize([scaler(dataset[:], undo=True), scaler(x_hat.detach().cpu().numpy(), undo=True)], projection='2d')
    # print(torch.softmax(l[:5], dim=-1))

    one_hot = np.eye(10).astype(np.float32)
    # print(one_hot.shape)

    x_hat, x_hat_var = vae.decoder(torch.tensor(one_hot).to(device))
    visualize([scaler(x_hat.detach().cpu().numpy(), undo=True)], projection='2d')