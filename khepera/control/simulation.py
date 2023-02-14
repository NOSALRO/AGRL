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
m = fastsim.Map("../data_collection/worlds/three_wall.pbm", 600)
datapoints = np.loadtxt("../data_collection/datasets/three_wall.dat")
vae = torch.load("../models/vae_three_wall.pt")
net = Net(3, 2, [128, 64]).to(device)
target = datapoints[5000]
x_hat, x_hat_var, latent, _ = vae(torch.tensor(target, device=device).float(), device, True, True)
distribution = MultivariateNormal(x_hat.cpu(), torch.diag(x_hat_var.mul(.5).exp().cpu()))

print(reward(torch.tensor(target), distribution))
res = cma.evolution_strategy.fmin(run, x0=get_model_params(net), sigma0=0.5, options={'CMA_elitist' : True, "maxfevals" : 2000}, args=(net, distribution, target, m, 0))

run(res[0], net, distribution, target, m, 2)
