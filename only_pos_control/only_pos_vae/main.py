import sys
import os
import torch
from torch.utils.data import DataLoader
import torch.functional as F

from data_loading import  StateData
from utils import vizualization, train, plot_latent_reconstructed
from vae_arch import VarAutoencoder

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StateData("../../data/archive_10000.dat")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True
    )

    # vizualization(dataset.get_data())

    vae = VarAutoencoder(3, 3, hidden_sizes = [256, 128]).to(device)

    epochs = 300
    lr = 5e-4
    vae = train(vae, epochs, lr, dataloader, device)
    torch.save(vae, "../models/vae_only_pos.pt")
    plot_latent_reconstructed(vae, len(dataset), device)

    i = torch.tensor(dataset.get_data()).to(device)
    out = vae.forward(i, deterministic=True)
    vizualization(dataset.get_data(), title="Original Data Points")
    vizualization(out[0].detach().cpu(), title="Reconstructed Data")
    vizualization(out[1].detach().cpu().exp(), title="Decoder Variance")
    # vizualization(out[2].detach().cpu())
    vizualization(out[3].detach().cpu().exp(), title="Encoder Variance")
