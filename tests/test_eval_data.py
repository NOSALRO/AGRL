import sys
import torch
from agrl.vae import StatesDataset, VariationalAutoencoder, train, visualize
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import numpy as np
import pyfastsim as fastsim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = StatesDataset(np.loadtxt(sys.argv[1]))
vae = torch.load(sys.argv[2])
i, x_hat_var, mu, logvar = vae(torch.tensor(dataset[900]).to(device), device, True, scale=True)
print(mu)
# visualize([dataset[:], i.detach().cpu().numpy()], projection='2d', file_name='.tmp/image')
