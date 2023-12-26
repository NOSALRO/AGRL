import torch
import numpy as np
from agrl.env import KheperaDVControllerEnv, Box
from agrl.controllers import DVController
from agrl.rl.td3 import Actor, Critic, train_td3, eval_policy
from agrl.vae import StatesDataset, VariationalAutoencoder, train, visualize
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
from agrl.rl.utils import train_sb3, cli
import pyfastsim as fastsim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Env Init.
world_map = fastsim.Map('worlds/dots.pbm', 600)
robot = fastsim.Robot(10, fastsim.Posture(500., 500., 0.))

action_space = Box(low=-1., high=1., shape=(2,), dtype=np.float32)
observation_space = Box(
    low=np.array([0, 0, -1, -1]),
    high=np.array([1, 1, 1, 1]),
    shape=(4,),
    dtype=np.float32
)
start_space = Box(
    low=np.array([25, 25, -np.pi]),
    high=np.array([575, 575, np.pi]),
    shape=(3,),
    dtype=np.float32
)

env = KheperaDVControllerEnv(
    robot=robot,
    world_map=world_map,
    reward_type='distance',
    n_obs=4,
    goals=[180, 180, 0],
    goal_conditioned_policy=False,
    latent_rep=False,
    observation_space=observation_space,
    action_space=action_space,
    random_start=False,
    max_steps=0,
    vae=None,
    scaler=None,
    controller=DVController(),
    sigma_sq=100
)

actor_net = Actor(observation_space.shape[0], action_space.shape[0], 1)
critic_net = Critic(observation_space.shape[0], action_space.shape[0])

train_td3(env, actor_net, critic_net)
