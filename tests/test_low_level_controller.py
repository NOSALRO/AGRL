import numpy as np
import torch
import gym
from nosalro.env import KheperaWithControllerEnv
from nosalro.vae import StatesDataset, VariationalAutoencoder, train, visualize
import pyfastsim as fastsim

# Env Init.
dataset = StatesDataset(path="data/no_wall.dat", angle_to_sin_cos=True, angle_column=2)
map = fastsim.Map('../goal-conditioned-skill-learning.bak/khepera/data_collection/worlds/no_wall.pbm', 600)
robot = fastsim.Robot(10, fastsim.Posture(50., 40., 0.))
action_space = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
observation_space = gym.spaces.Box(
    low=np.array([0, 0, -1, -1, -np.inf, -np.inf]),
    high=np.array([600, 600, 1, 1, np.inf, np.inf]),
    shape=(6,),
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
    map=map,
    reward_type='mse',
    target=[500, 400, 0],
    n_obs=4,
    goals=None,
    goal_conditioned_policy=False,
    latent_rep=False,
    observation_space=observation_space,
    action_space=action_space,
    random_start=start_space,
    max_steps=1000,
)

env.reset()
env.render()
while True:
    env.step(dataset.get_data()[100][:2])
    env.reset()