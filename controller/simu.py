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
    return dist.log_prob(observations).detach().item()

def distance_reward_fn(position, target):
    return np.linalg.norm(position - target, ord=2)

def get_model_params(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy()

def set_model_params(model, params):
    torch.nn.utils.vector_to_parameters(torch.Tensor(params), model.parameters())

def run_simu(model_params, dt, dist=None, target=None, disp=False):
    net = Net(12, 3, [128, 64, 32], action_range=3.)
    set_model_params(net, model_params)

    MASS = 1.#30.4213964625
    INERTIA = [1., 1., 1., 0., 0., 0.] #[0.88201174, 1.85452968, 1.97309185, 0.00137526, 0.00062895, 0.00018922]
    robot = Robot(MASS, INERTIA)
    robot.set_initial_position(np.array([0., 0., 0.]))
    robot.set_initial_orientation(np.array([0., 0., 0.]))

    reward = 0
    for _ in range(200):
        observations = torch.tensor(robot.state()).float().view(1, -1)
        action = net(observations)
        action = action.detach().numpy()[0]
        # robot.step(action[0].reshape(3,1), action[1].reshape(3,1), dt)
        robot.step(action, np.zeros(3), dt)

        if disp:
            # print(f"Robot Pos -> {robot.get_position().reshape(3,)}, Action -> {action[0].reshape(3,)}, {action[1].reshape(3,)}, Target -> {target}")
            print(f"Robot Pos -> {robot.get_position()}, Target -> {target.detach().numpy()}, Action -> {action.reshape(3,)}")
        reward += edl_reward_fn(observations, dist)
        # reward += distance_reward_fn(robot.get_position(), target)
    if disp:
        print(f"Reward -> {reward}, Robot Pos -> {robot.get_position()}, Target -> {target.detach().numpy()}")
    return -reward

net = Net(12, 3, [128, 64, 32], action_range=3.)

# random_target = np.random.uniform(low=-5., high=5., size=(3,))
# print("Target -> {random_target}")

vae = torch.load("../models/vae.pt")
vae = vae.cpu()
dataset = StateData("../data/archive_10000.dat")
random_state = dataset.get_data()[0]
target = vae.encoder(torch.tensor(random_state))[0]
mu, log_var = vae.decoder(target)
dist = MultivariateNormal(mu, torch.diag(torch.exp(0.5 * log_var)))

# scipy.optimize.minimize(run_simu, x0=get_model_params(net), args=(rob, net, random_target, 0.01), options={"maxiter": 10})

res = cma.evolution_strategy.fmin(run_simu, x0=get_model_params(net), sigma0=0.8, options={'CMA_elitist' : True, "maxfevals" : 2000, "tolflatfitness" : 5000}, args=(0.05, dist, target))

run_simu(res[0], 0.01, dist=dist, target=target, disp=True)
