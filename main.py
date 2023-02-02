import sys
import os
import torch
from torch.utils.data import DataLoader
import torch.functional as F

from data_loading import  StateData
from utils import vizualization, train, plot_latent_reconstructed
from vae_arch import VarAutoencoder

if __name__ == "__main__":

    device = torch.device("cuda" if True else "cpu")
    dataset = StateData("data/archive_10000.dat")

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=16, 
        shuffle=True
    )
    
    vizualization(dataset.get_data()) 

    vae = VarAutoencoder(12, 3).to(device)

    vae = train(vae, 10, dataloader, device)
    plot_latent_reconstructed(vae, device)