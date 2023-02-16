import sys
import torch
import pyfastsim as fastsim
import numpy as np
from torch.distributions import MultivariateNormal
from utils import get_model_params, set_model_params, reward, run
sys.path.append('../vae/')
from vae_arch import VAE, VariationalDecoder, VariationalEncoder

net = torch.load("../models/control_model.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = torch.load("../models/vae_three_wall.pt")
datapoints = np.loadtxt("../data_collection/datasets/three_wall.dat")
target = datapoints[3000]
x_hat, x_hat_var, latent, _ = vae(torch.tensor(target, device=device).float(), device, True, True)
print(x_hat, x_hat_var)
distribution = MultivariateNormal(x_hat.cpu(), torch.diag(x_hat_var.mul(.5).exp().cpu()))

run(get_model_params(net), net, distribution, target, vae.min, vae.max, 2)
