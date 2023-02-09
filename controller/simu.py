import copy
import sys
import torch
import numpy as np
from torch.distributions import MultivariateNormal
from robot import Robot
from utils import get_model_params, set_model_params, reward, run
from net import Net
import cma

sys.path.append('../')
from vae_arch import VAE, VariationalDecoder, VariationalEncoder

rob = Robot()
rob.set_position(np.array([0., 0., 0.]))
datapoints = np.loadtxt("../data/archive_10000.dat")
vae = torch.load("../models/vae.pt")
net = Net(12, 12, [128, 64, 32])
target = datapoints[2]
x_hat, x_hat_var, latent, _ = vae(torch.tensor(target).float(), True)
distribution = MultivariateNormal(x_hat, torch.diag(x_hat_var.mul(.5).exp()))

print(reward(torch.tensor(target), distribution))

res = cma.evolution_strategy.fmin(run, x0=get_model_params(net), sigma0=0.7, options={'CMA_elitist' : True, "maxfevals" : 5000}, args=(rob, net, distribution, target, 0))
run(res[0], rob, net, distribution, target, 2)
