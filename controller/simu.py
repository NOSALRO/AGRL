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

def reward_fn(position, target, dist=None):
    # reward = dist.log_prob(observations)
    reward = np.linalg.norm(position.reshape(1,-1)[0] - target, ord=2)
    return reward

def get_model_params(model):
    params = []
    with torch.no_grad():
        for i in model.parameters():
            params.append(i.detach().numpy())
    return params

def set_model_params(model, params):
    pass

def run_simu_once(model_params, robot, net, target, dist, dt, reward):
    # set_model_params(net, model_params)
    observations = torch.tensor(robot.state()).float().view(1, -1)
    action = net(observations)
    action = action.detach().numpy()
    robot.step(action[0].reshape(3,1), action[1].reshape(3,1), dt)
    reward += reward_fn(robot.get_position(), target)
    print(reward)
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
target = np.random.random(3) + 10

reward = 0
model_params = get_model_params(net)

scipy.optimize.minimize(run_simu_once, x0=model_params, args=(model_params, rob, net, target, dist, 0.1, reward))
