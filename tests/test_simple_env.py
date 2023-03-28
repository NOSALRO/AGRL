import numpy as np
import torch
import gym
from nosalro.env import SimpleEnv
from nosalro.robots import Point
from nosalro.vae import StatesDataset, VariationalAutoencoder, train, visualize
from nosalro.transforms import Scaler, Shuffle, Compose

device = torch.device("mps" if torch.cuda.is_available() else "cpu")
robot = Point([10, 10])
shuffle = Shuffle(seed=42)
scaler = Scaler()

transforms = Compose([
    shuffle,
    scaler.fit,
    scaler
    ])

dataset = StatesDataset(path="data/random_points.dat", transforms=transforms)

vae = VariationalAutoencoder(2, 2, output_dims=2, scaler=scaler)
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
    n_obs=2,
    goals=[20,20],#dataset.get_data()[[8000, 200]],
    goal_conditioned_policy=True,
    latent_rep=False,
    observation_space=observation_space,
    action_space=action_space,
    random_start=False,
    max_steps=100,
    vae=vae,
    scaler=scaler,
)

env.reset()
for _ in range(10): print(env.step([10, 0]))