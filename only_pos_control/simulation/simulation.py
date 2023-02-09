import copy
import sys
import torch
import numpy as np
from torch.distributions import MultivariateNormal
from robot import Robot
from utils import get_model_params, set_model_params, reward, run
from net import Net
import cma

sys.path.append('../vae/')
from vae_arch import VAE, VariationalDecoder, VariationalEncoder

rob = Robot()
rob.set_position(np.array([0., 0., 0.]))
datapoints = np.loadtxt("../../data/archive_10000.dat")
vae = torch.load("../models/vae_only_pos.pt")
net = Net(3, 3, [32])
target = datapoints[0, :3]
x_hat, x_hat_var, latent, _ = vae(torch.tensor(target).float(), 'cpu', True)
distribution = MultivariateNormal(x_hat, torch.diag(x_hat_var.mul(.5).exp()))

print(reward(torch.tensor(target), distribution))


res = cma.evolution_strategy.fmin(run, x0=get_model_params(net), sigma0=0.5, options={'CMA_elitist' : True, "maxfevals" : 5000}, args=(rob, net, distribution, target, 0))

run(res[0], rob, net, distribution, target, 2)
