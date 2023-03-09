import numpy as np
import torch
import gym
from nosalro.env import KheperaEnv
from nosalro.vae import StatesDataset, VariationalAutoencoder, train, visualize
import pyfastsim as fastsim

# VAE Train.
dataset = StatesDataset(path="data/no_wall.dat", angle_to_sin_cos=True, angle_column=2)
_min, _max = dataset.scale_data()
vae_arch = VariationalAutoencoder(4, 2, min=_min, max=_max, hidden_sizes=[256, 128])
epochs = 300
lr = 1e-4
vae = train(
    vae_arch,
    epochs,
    lr,
    dataset,
    'cpu',
    beta = 1e-4,
    file_name='vae_no_wall.pt',
    overwrite=False,
    weight_decay=0,
    batch_size = 512,
)
# visualize(dataset.get_data(), projection='2d', columns_select=[0,1])
# visualize(vae(torch.tensor(dataset.get_data()), 'cpu', True, True)[0].detach().cpu().numpy(), projection='2d')

# Env Init.
map = fastsim.Map('../goal-conditioned-skill-learning.bak/khepera/data_collection/worlds/no_wall.pbm', 600)
robot = fastsim.Robot(10, fastsim.Posture(100., 100., 0.))
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

env = KheperaEnv(
    robot=robot,
    map=map,
    reward_type='mse',
    target=False,
    n_obs=4,
    goals=dataset.get_data(),
    goal_conditioned_policy=True,
    latent_rep=True,
    observation_space=observation_space,
    action_space=action_space,
    random_start=start_space,
    max_steps=100,
    vae=vae,
    min_max=[vae.min, vae.max]
)

# Agent Act.
env.reset()
env.render()
for i in range(10000):
    if not (i % 100):
        env.reset()
        env.close()
        env.render()
    env.step([10, 10.1])