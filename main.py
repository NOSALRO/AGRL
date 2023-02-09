import sys
import os
import torch
from torch.utils.data import DataLoader

from data_loading import  StateData
from utils import vizualization, train, plot_latent_reconstructed
from vae_arch import VAE, VariationalEncoder, VariationalDecoder

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StateData("data/archive_10000.dat")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True
    )

    # vizualization(dataset.get_data())

    vae = VAE(12, 3, hidden_sizes = [512, 256]).to(device)

    epochs = 500
    lr = 5e-4
    vae = train(vae, epochs, lr, dataloader, device, beta = 0.1)
    # plot_latent_reconstructed(vae, len(dataset), device)

    i = torch.tensor(dataset.get_data()).to(device)
    x_hat, x_hat_var, latent, _ = vae(i, device, deterministic=True)
    vizualization(dataset.get_data())
    vizualization(x_hat.detach().cpu())
    # vizualization(out[1].detach().cpu().exp())
    # vizualization(out[2].detach().cpu())
    # vizualization(out[3].detach().cpu().exp())
    torch.save(vae, "models/vae.pt")
