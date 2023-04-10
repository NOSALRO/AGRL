import numpy as np
import torch
import gym
from nosalro.env import KheperaWithControllerEnv
from nosalro.vae import StatesDataset, VariationalAutoencoder, train, visualize
from nosalro.es import cma_run, env_run, cma_eval, PolicyNet
import pyfastsim as fastsim

net = PolicyNet(
        input_dim=4,
        output_dim=2,
        hidden_layers=[(16, torch.nn.ReLU()), (16, torch.nn.ReLU())],
        output_activation=torch.nn.Tanh(),
    )

dataset = StatesDataset(path="data/no_wall.dat")
world_map = fastsim.Map('worlds/no_wall.pbm', 600)
robot = fastsim.Robot(10, fastsim.Posture(50., 40., np.pi))
action_space = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
observation_space = gym.spaces.Box(
    low=np.array([0, 0, -1, -1]),
    high=np.array([600, 600, 1, 1]),
    shape=(4,),
    dtype=np.float32
)

start_space = gym.spaces.Box(
    low=np.array([50, 50, -np.pi]),
    high=np.array([550, 550, np.pi]),
    shape=(3,),
    dtype=np.float32
)

env = KheperaWithControllerEnv(
    robot=robot,
    world_map=world_map,
    reward_type='mse',
    target=None,
    n_obs=4,
    goals=dataset[[100]],
    goal_conditioned_policy=False,
    latent_rep=False,
    observation_space=observation_space,
    action_space=action_space,
    random_start=start_space,
    max_steps=60,
)

best_sol = cma_run(env, net)
cma_eval(env, net, best_sol)

## best_sol = cma_run(env, net)
#net = torch.load('models/policies/cma_policy.pt')
#cma_eval(env, net, None)