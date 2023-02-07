import sys
import numpy as np
import torch
from torch.distributions import MultivariateNormal
import cma
import scipy

from robot import Robot
from net import Net

sys.path.append("./only_pos_vae/")
from data_loading import StateData
from vae_arch import VarAutoencoder

def edl_reward_fn(observations, dist):
    return dist.log_prob(observations).detach().item()

def distance_reward_fn(position, target):
    return np.linalg.norm(position - target, ord=2)

def get_model_params(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy()

def set_model_params(model, params):
    torch.nn.utils.vector_to_parameters(torch.Tensor(params), model.parameters())

def run_simu(model_params, dt, dist=None, target=None, disp=False):
    net = Net(3, 3, [128, 64, 32], action_range=4.)
    set_model_params(net, model_params)

    MASS = 1.#30.4213964625
    INERTIA = [1., 1., 1., 0., 0., 0.] #[0.88201174, 1.85452968, 1.97309185, 0.00137526, 0.00062895, 0.00018922]
    robot = Robot(MASS, INERTIA)
    robot.set_initial_position(np.array([0., 0., 0.]))
    robot.set_initial_orientation(np.array([0., 0., 0.]))

    reward = 0
    for _ in range(2):
        observations = torch.tensor(robot.state()).float().view(1, -1)
        action = net(observations)
        action = action.detach().numpy()[0]
        robot.step(action)

        if disp:
            print(f"Robot Pos -> {robot.get_position()}, Target -> {target.detach().numpy()}, Action -> {action.reshape(3,)}")
        reward += edl_reward_fn(observations, dist)
    if disp:
        print(f"Reward -> {reward}, Robot Pos -> {robot.get_position()}, Target -> {target.detach().numpy()}")
    return -reward

net = Net(3, 3, [128, 64, 32], action_range=4.)
vae = torch.load("models/vae_only_pos.pt")
vae = vae.cpu()
dataset = StateData("../data/archive_10000.dat")
random_state = dataset.get_data()[0, :3]
target = vae.encoder(torch.tensor(random_state))[0]
mu, log_var = vae.decoder(target)
dist = MultivariateNormal(mu, torch.diag(torch.exp(0.5 * log_var)))

res = cma.evolution_strategy.fmin(run_simu, x0=get_model_params(net), sigma0=0.5, options={'CMA_elitist' : True, "maxfevals" : 2000, "tolflatfitness" : 5000}, args=(0.05, dist, target, False))

run_simu(res[0], 0.01, dist=dist, target=target, disp=True)
