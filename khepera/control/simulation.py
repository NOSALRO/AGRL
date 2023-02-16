import copy, math
import sys
import torch
import numpy as np
from torch.distributions import MultivariateNormal
from utils import get_model_params, set_model_params, reward, run
from net import Net
import pyfastsim as fastsim
import cma

sys.path.append('../vae/')
from vae_arch import VAE, VariationalDecoder, VariationalEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datapoints = np.loadtxt("../data_collection/datasets/three_wall.dat")
vae = torch.load("../models/vae_three_wall.pt")
net = Net(3, 2, [128, 64]).to(device)
target = datapoints[10000]
x_hat, x_hat_var, latent, _ = vae(torch.tensor(target, device=device).float(), device, True, True)
print(x_hat, x_hat_var)
distribution = MultivariateNormal(x_hat.cpu(), torch.diag(x_hat_var.mul(.5).exp().cpu()))

print(reward((torch.tensor(target) - vae.min)/(vae.max-vae.min), distribution))
print(target)
res = cma.evolution_strategy.fmin(run, x0=get_model_params(net), sigma0=0.4, options={'CMA_elitist' : True, "maxfevals" : 1000}, args=(net, distribution, target, vae.min, vae.max, 0))

run(res[0], net, distribution, target, vae.min, vae.max, 2)
set_model_params(net, res[0])
torch.save(net, "../models/control_model.pt")