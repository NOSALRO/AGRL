import numpy as np
import torch
import gym
from nosalro.env import SimpleEnv
from nosalro.robots import Point
from nosalro.vae import StatesDataset, VariationalAutoencoder, train, visualize
from nosalro.rl import learn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = StatesDataset(path="data/pdp_data.dat")
_min, _max = dataset.scale_data()
vae = VariationalAutoencoder(2, 2, min=_min, max=_max)
epochs = 100
lr = 5e-4
vae = train(
    vae,
    epochs,
    lr,
    dataset,
    device,
    beta = 0.1,
    file_name='models/vae_models/vae_pdp.pt',
    overwrite=False,
    weight_decay=0
)
# visualize(dataset.get_data(), projection='2d')
# visualize(vae(torch.tensor(dataset.get_data()), 'cpu', True, False)[0].detach().cpu().numpy(), projection='2d')

robot = Point([10, 10])
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
    target=None,
    n_obs=2,
    goals=dataset.get_data()[[8000, 200]],
    goal_conditioned_policy=True,
    latent_rep=True,
    observation_space=observation_space,
    action_space=action_space,
    random_start=True,
    max_steps=0,
    vae=vae,
    min_max=[vae.min, vae.max]
)
learn(env, device)