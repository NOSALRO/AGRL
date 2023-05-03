import os
import copy
from nosalro.vae import StatesDataset
from nosalro.transforms import Compose, AngleToSinCos, Scaler, Shuffle
from nosalro.vae import VQVAE
import numpy as np
import torch



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = Scaler()
    shuffler = Shuffle()
    transforms = Compose([shuffler, scaler.fit, scaler])
    dataset = StatesDataset(path='data/alley_right_go_explore.dat', transforms=transforms)

    vqvae = VQVAE(input_dims=2, latent_dims=8, codes=10, output_dims=2, beta=1.25)
    vqvae = vqvae.to(device)

    vqvae = vqvae.train(
        loss_fn=vqvae.loss_fn,
        model=vqvae,
        epochs=1000,
        lr=3e-4,
        dataset = dataset,
        device = device,
        overwrite=False,
        file_name = '.tmp/vqvae_test',
        batch_size = 128
    )

    x_hat, z, _ = vqvae(torch.tensor(dataset[:], device=device), device, False, False)
    print(np.unique(z.detach().cpu().numpy(), axis=-1).shape)
