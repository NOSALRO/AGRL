import numpy as np
import torch
import gym
from nosalro.env import SimpleEnv
from nosalro.robots import Point
from nosalro.vae import StatesDataset, VariationalAutoencoder, Scaler, train, visualize
from nosalro.rl import learn

dataset = StatesDataset(path="data/random_points.dat")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = Scaler('standard')
scaler.fit(dataset.get_data())
dataset.scale_data(scaler)
vae_arch = VariationalAutoencoder(2, 2, scaler=scaler, hidden_sizes=[128, 64]).to(device)
epochs = 500
lr = 3e-4
vae = train(
    vae_arch,
    epochs,
    lr,
    dataset,
    device,
    beta = 1,
    file_name='models/vae_models/vae_random_points_pdp.pt',
    overwrite=False,
    weight_decay=0,
    batch_size = 128,
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
    reward_type='edl',
    target=None,
    n_obs=2,
    goals=dataset.get_data(),
    goal_conditioned_policy=True,
    latent_rep=True,
    observation_space=observation_space,
    action_space=action_space,
    random_start=True,
    max_steps=0,
    vae=vae,
    scaler=scaler
)
learn(env, device)