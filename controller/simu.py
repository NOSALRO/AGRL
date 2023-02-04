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

def reward_fn(observations, dist):
    reward = dist.log_prob(observations).detach().item()
    # reward = np.linalg.norm(position.reshape(1,-1)[0] - target, ord=2)
    return reward

def get_model_params(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy()

def set_model_params(model, params):
    torch.nn.utils.vector_to_parameters(torch.Tensor(params), model.parameters())

def run_simu_once(model_params, robot, net, target, dist, dt, reward):
    set_model_params(net, model_params)
    _r = 0
    for _ in range(30):
        observations = torch.tensor(robot.state()).float().view(1, -1)
        action = net(observations)
        action = action.detach().numpy()
        robot.step(action[0].reshape(3,1), action[1].reshape(3,1), dt)
        _r += -reward_fn(observations, dist)
    reward = 0.9*reward + 0.1*_r
    print(f"Reward -> {reward}, Robot Pos -> {robot.get_position().reshape(3,)}, Target -> {target.detach().numpy().reshape(3,)}")
    return reward


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

reward = 0

scipy.optimize.minimize(run_simu_once, x0=get_model_params(net), args=(rob, net, target, dist, 0.01, reward))
