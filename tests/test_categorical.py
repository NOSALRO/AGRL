import sys
import os
import random
import torch
import numpy as np
from nosalro.vae import CategoricalVAE, StatesDataset, train, visualize, categorical_train
from nosalro.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shuffle = Shuffle(seed=42)
    scaler = Scaler()
    transforms = Compose([shuffle, scaler.fit, scaler])

    dataset = StatesDataset(path='data/alley_right_go_explore.dat', transforms=transforms)

    cvae = CategoricalVAE(2, 100, output_dims=2, hidden_sizes=[128,128], scaler=scaler).to(device)
    epochs = 1000
    lr = 7e-04
    vae = categorical_train(
        cvae,
        epochs,
        lr,
        dataset,
        device,
        beta = 100,
        file_name = '.tmp/categorical_vae.pt',
        overwrite = True,
        weight_decay = 0,
        batch_size = 32,
    )

    x_hat, x_hat_var, l = vae(torch.tensor(dataset[:]).to(device), device, True, False)
    visualize([scaler(dataset[:], undo=True), scaler(x_hat.detach().cpu().numpy(), undo=True)], projection='2d')

    # l = l.detach().cpu().numpy()
    # one_hot = np.zeros(l.shape[-1], )
    # one_hot[np.argmax(l[1])] = 1
    # print(one_hot)