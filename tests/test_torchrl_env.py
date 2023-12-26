import time
import numpy as np
import torch
# from agrl.env import Box
from agrl.env.torchrl_khepera import KheperaEnv
from agrl.env.torchrl_base import BaseEnv
from agrl.vae import StatesDataset, VariationalAutoencoder, train, visualize
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import pyfastsim as fastsim
import gymnasium as gym
import torchrl

if __name__ == '__main__':
    dataset = StatesDataset(path="data/no_wall.dat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Env Init.
    world_map = fastsim.Map('worlds/no_wall.pbm', 600)
    robot = fastsim.Robot(10, fastsim.Posture(100., 100., 0))

    action_space = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
    observation_space = gym.spaces.Box(
        low=np.array([0, 0, -1, -1]),
        high=np.array([600, 600, 1, 1]),
        shape=(4,),
        dtype=np.float32
    )
    start_space = gym.spaces.Box(
        low=np.array([25, 25, -np.pi]),
        high=np.array([575, 575, np.pi]),
        shape=(3,),
        dtype=np.float32
    )

    env = KheperaEnv(
        robot=robot,
        world_map=world_map,
        reward_type='mse',
        n_obs=3,
        goals=[500, 200, 0],
        goal_conditioned_policy=False,
        latent_rep=False,
        observation_space=observation_space,
        action_space=action_space,
        random_start=False,
        max_steps=1000,
        vae=None,
        scaler=None
    )
    print(env._reset())
    torchrl.envs.utils.check_env_specs(env)
