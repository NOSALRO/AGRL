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
rob.set_position(np.array([0., 0.]))
datapoints = np.load("../vae/random_data.npy")
vae = torch.load("../models/vae.pt")
net = Net(2, 2, [128])
target = datapoints[1]
x_hat, x_hat_var, latent, _ = vae(torch.tensor(target).float(), 'cpu', True)
distribution = MultivariateNormal(x_hat, torch.diag(x_hat_var.mul(.5).exp()))


res = cma.evolution_strategy.fmin(run, x0=get_model_params(net), sigma0=0.5, options={'CMA_elitist' : True, "maxfevals" : 10000, "tolflatfitness" : 5000}, args=(rob, net, distribution, target, 0))

run(res[0], rob, net, distribution, target, 1)