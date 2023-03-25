import numpy as np
import torch
import gym
from nosalro.env import KheperaControllerEnv, Box
from nosalro.vae import StatesDataset, VariationalAutoencoder, Scaler, train, visualize
from nosalro.rl import learn
from nosalro.controllers import Controller
import pyfastsim as fastsim

if __name__ == '__main__':
    dataset = StatesDataset(path="data/no_wall.dat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Env Init.
    world_map = fastsim.Map('worlds/no_wall.pbm', 600)
    robot = fastsim.Robot(10, fastsim.Posture(100., 100., 0.))

    action_space = Box(low=-1., high=1., shape=(2,), dtype=np.float32)
    observation_space = Box(
        low=np.array([0, 0, -1, -1]),
        high=np.array([600, 600, 1, 1]),
        shape=(4,),
        dtype=np.float32
    )
    start_space = Box(
        low=np.array([25, 25, -np.pi]),
        high=np.array([575, 575, np.pi]),
        shape=(3,),
        dtype=np.float32
    )
    env = KheperaControllerEnv(
        robot=robot,
        world_map=world_map,
        reward_type='mse',
        n_obs=4,
        goals=[500, 200, 0],
        goal_conditioned_policy=False,
        latent_rep=False,
        observation_space=observation_space,
        action_space=action_space,
        random_start=start_space,
        max_steps=0,
        vae=None,
        scaler=None,
        controller=Controller(0.5, 0)
    )

    # # Agent Train.
    learn(env, device)
