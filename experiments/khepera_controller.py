import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
from nosalro.env import KheperaWithControllerEnv
from nosalro.vae import StatesDataset, VariationalAutoencoder, Scaler, train, visualize
from nosalro.rl import learn
import pyfastsim as fastsim

if __name__ == '__main__':
    # VAE Train.
    dataset = StatesDataset(path="data/no_wall.dat", angle_to_sin_cos=True, angle_column=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Env Init.
    map = fastsim.Map('worlds/no_wall.pbm', 600)
    robot = fastsim.Robot(10, fastsim.Posture(100., 100., 0.))

    action_space = gym.spaces.Box(low=50., high=550., shape=(2,), dtype=np.float32)
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
    map=map,
    reward_type='mse',
    target=False,
    n_obs=4,
    goals=dataset.get_data()[[100]],
    goal_conditioned_policy=False,
    latent_rep=False,
    observation_space=observation_space,
    action_space=action_space,
    random_start=start_space,
    max_steps=0,
    vae=None,
    scaler=None
    )

    # # Agent Train.
    learn(env, device)
