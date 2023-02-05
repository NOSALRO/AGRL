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

def edl_reward_fn(observations, dist):
    reward = dist.log_prob(observations).detach().item()
    return reward

def distance_reward_fn(position, target):
    reward = np.linalg.norm(position.reshape(1,-1)[0] - target, ord=2)
    return reward

def get_model_params(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy()

def set_model_params(model, params):
    torch.nn.utils.vector_to_parameters(torch.Tensor(params), model.parameters())

def run_simu(model_params, robot, net, target, dt, dist=None):
    set_model_params(net, model_params)
    reward = 0
    for _ in range(50):
        observations = torch.tensor(robot.state()).float().view(1, -1)
        action = net(observations)
        action = action.detach().numpy()
        robot.step(action[0].reshape(3,1), action[1].reshape(3,1), dt)
        # reward += edl_reward_fn(observations, dist)
        reward += -distance_reward_fn(robot.get_position(), target)
    print(f"Reward -> {reward/50}, Robot Pos -> {robot.get_position().reshape(3,)}, Target -> {target}")
    return distance_reward_fn(robot.get_position(), target)



MASS = 30.4213964625
INERTIA = [0.88201174, 1.85452968, 1.97309185, 0.00137526, 0.00062895, 0.00018922]
rob = Robot(MASS, INERTIA)
rob.set_initial_position(np.array([0., 0., 0.]).reshape(-1, 1))
rob.set_initial_orientation(np.array([0., 0., 0.]).reshape(-1, 1))

net = Net(12, 6, [128, 64])

random_target = np.random.uniform(low=1, high=10, size=(3,))

# scipy.optimize.minimize(run_simu, x0=get_model_params(net), args=(rob, net, random_target, 0.01), options={"maxiter": 10})

cma.fmin(run_simu, x0=get_model_params(net), sigma0=1, args=(rob, net, random_target, 0.01))



# vae = torch.load("../models/vae.pt")
# vae = vae.cpu()
# dataset = StateData("../data/archive_10000.dat")
# random_state = dataset.get_data()[0]
# target = vae.encoder(torch.tensor(random_state))[0]
# mu, log_var = vae.decoder(target)
# dist = MultivariateNormal(mu, torch.diag(torch.exp(0.5 * log_var)))