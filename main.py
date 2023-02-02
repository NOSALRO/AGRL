import sys
import os
import torch
from torch.utils.data import DataLoader
import torch.functional as F

from data_loading import  StateData
from utils import vizualization, train, plot_latent, plot_reconstructed
from vae_arch import VarAutoencoder

if __name__ == "__main__":

    dataset = StateData("data/archive_10000.dat")

    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=32, 
        shuffle=True
    )
    
    vizualization(dataset.get_data()) 

    vae = VarAutoencoder(12, 3)

    vae = train(vae, 100, dataloader)
    plot_latent(vae, dataloader)
    plot_reconstructed(vae, dataloader)