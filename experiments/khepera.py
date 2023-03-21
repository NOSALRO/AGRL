import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
from nosalro.env import KheperaEnv
from nosalro.vae import StatesDataset, VariationalAutoencoder, Scaler, train, visualize
from nosalro.rl import learn
import pyfastsim as fastsim

# VAE Train.
dataset = StatesDataset(path="data/no_wall.dat", angle_to_sin_cos=True, angle_column=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = Scaler('standard')
scaler.fit(dataset.get_data())
dataset.scale_data(scaler)
vae_arch = VariationalAutoencoder(4, 2, scaler=scaler, hidden_sizes=[256, 128]).to(device)
epochs = 300
lr = 1e-4
vae = train(
    vae_arch,
    epochs,
    lr,
    dataset,
    device,
    beta = 2,
    file_name='models/vae_models/vae_no_wall.pt',
    overwrite=False,
    weight_decay=0,
    batch_size = 512,
)
# visualize(dataset.get_data(), projection='2d')
# visualize(vae(torch.tensor(dataset.get_data()), device, True, False)[0].detach().cpu().numpy(), projection='2d')

vae = vae.to(device)
# Env Init.
world_map = fastsim.Map('worlds/no_wall.pbm', 600)
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
    world_map=world_map,
    reward_type='mse',
    target=False,
    n_obs=4,
    goals=dataset.get_data()[[0, -1]],
    goal_conditioned_policy=True,
    latent_rep=True,
    observation_space=observation_space,
    action_space=action_space,
    random_start=start_space,
    max_steps=0,
    vae=vae,
    scaler=scaler
)

# # Agent Train.
learn(env, device)
