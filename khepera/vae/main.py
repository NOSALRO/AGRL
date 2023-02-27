import sys
import os
import torch
from torch.utils.data import DataLoader

from data_loading import  ObstacleData
from utils import vizualization, train, plot_latent_reconstructed
from vae_arch import VAE

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ObstacleData("../data_collection/datasets/no_wall.dat")
    _min, _max = dataset.scale_data()

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=512,
        shuffle=True
    )

    vae = VAE(4, 2, _min, _max, hidden_sizes = [256, 128]).to(device)

    epochs = 500
    lr = 1e-4
    vae = train(vae, epochs, lr, dataloader, device)
    torch.save(vae, "../models/vae_no_wall_cos_sin.pt")
    # vae = torch.load("../models/vae_three_wall.pt")
    # plot_latent_reconstructed(vae, len(dataset), device)

    i = torch.tensor(dataset.get_data()).to(device)
    out = vae(i, device, deterministic=True)
    vizualization(dataset.get_data(), title="Original Data Points")
    vizualization(out[0].detach().cpu(), title="Reconstructed Data")
    # vizualization(out[1].detach().cpu().exp(), title="Decoder Variance")
    # vizualization(out[2].detach().cpu())
    # vizualization(out[3].detach().cpu().exp(), title="Encoder Variance")