import sys
import numpy as np
import torch
from torch.distributions import MultivariateNormal
import cma
import scipy

from robot import Robot
from net import Net

sys.path.append("../")
from data_loading import StateData
from vae_arch import VarAutoencoder

def opt(net, reward):
    pass

def run_simu(robot, net, target, dt):
    reward = 0
    robot.step(np.zeros((3,1), dtype=np.float32), np.zeros((3,1), np.float32), dt)
    for _ in range(30):
        observations = torch.tensor(robot.state()).float().view(1, -1)
        reward += dist.log_prob(observations)
        opt(net, reward)
        action = net(observations)
        action = action.detach().numpy()
        robot.step(action[0].reshape(3,1), action[1].reshape(3,1), dt)


MASS = 8
INERTIA = np.random.random((3, 3))
rob = Robot(MASS, INERTIA)
rob.set_initial_position(np.array([0., 0., 0.]).reshape(-1, 1))
rob.set_initial_orientation(np.array([0., 0., 0.]).reshape(-1, 1))

net = Net(12, 6)
vae = torch.load("../models/vae.pt")
vae = vae.cpu()
dataset = StateData("../data/archive_10000.dat")
random_state = dataset.get_data()[0]
target = vae.encoder(torch.tensor(random_state))[0]
mu, log_var = vae.decoder(target)
dist = MultivariateNormal(mu, torch.diag(torch.exp(0.5 * log_var)))

run_simu(rob, net, target, 0.1)
