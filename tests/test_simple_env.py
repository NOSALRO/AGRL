import numpy as np
import torch
import gym
from nosalro.env import SimpleEnv
from nosalro.robots import Point
from nosalro.vae import StatesDataset, VariationalAutoencoder, train, visualize

robot = Point([10, 10])
dataset = StatesDataset(path="data/data.dat", angle_to_sin_cos=False, angle_column=2)
_min, _max = dataset.scale_data()

vae = VariationalAutoencoder(2, 2, scale=True, min=_min, max=_max)
epochs = 100
lr = 5e-4
vae = train(
    vae,
    epochs,
    lr,
    dataset,
    'cpu',
    beta = 0.1,
    file_name='vae_pdp.pt',
    overwrite=False,
    weight_decay=0
)
# visualize(dataset.get_data(), projection='2d')
# visualize(vae(torch.tensor(dataset.get_data()), 'cpu', True, False)[0].detach().cpu().numpy(), projection='2d')
action_space = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
observation_space = gym.spaces.Box(
    low=np.array([0, 0, -np.inf, -np.inf]),
    high=np.array([600, 600, np.inf, np.inf]),
    shape=(4,),
    dtype=np.float32
)

env = SimpleEnv(
    robot=robot,
    reward_type='mse',
    target=[20,20],
    n_obs=2,
    goals=None,#dataset.get_data()[[8000, 200]],
    goal_conditioned_policy=True,
    latent_rep=False,
    observation_space=observation_space,
    action_space=action_space,
    random_start=False,
    max_steps=100,
    vae=vae,
    min_max=[vae.min, vae.max]
)

env.reset()
for _ in range(10): print(env.step([10, 0]))