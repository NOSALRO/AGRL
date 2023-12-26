import sys
import numpy as np
import torch
import gym
from agrl.env import KheperaControllerEnv, Box
from agrl.controllers import Controller
from agrl.vae import StatesDataset, VariationalAutoencoder, train, visualize
import pyfastsim as fastsim

# Env Init.
dataset = StatesDataset(path="data/no_wall.dat")
world_map = fastsim.Map('worlds/no_wall.pbm', 600)
robot = fastsim.Robot(10, fastsim.Posture(500., 200., np.pi))

action_space = Box(low=-1., high=1., shape=(2,), dtype=np.float32)

observation_space = Box(
    low=np.array([0, 0, -1, -1]),
    high=np.array([600, 600, 1, 1]),
    shape=(4,),
    dtype=np.float32
)

start_space = Box(
    low=np.array([50, 50, -np.pi]),
    high=np.array([550, 550, np.pi]),
    shape=(3,),
    dtype=np.float32
)

env = KheperaControllerEnv(
    robot=robot,
    world_map=world_map,
    reward_type='mse',
    n_obs=4,
    goals=[500, 200, np.pi],
    goal_conditioned_policy=False,
    latent_rep=False,
    observation_space=observation_space,
    action_space=action_space,
    random_start=start_space,
    max_steps=50,
    controller=Controller(0.5, 0)
)

env.reset()
env.render()
while True:
    res = env.step(observation_space.scale(np.array(env.goals[0][:2]), -1, 1))
    if res[-2]:
        env.reset()