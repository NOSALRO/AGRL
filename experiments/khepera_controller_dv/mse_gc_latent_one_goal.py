import torch
import numpy as np
from nosalro.env import KheperaDVControllerEnv, Box
from nosalro.controllers import DVController
from nosalro.rl.td3 import Actor, Critic, train_td3, eval_policy
from nosalro.vae import StatesDataset, VariationalAutoencoder, train, visualize
from nosalro.transforms import Compose, AngleToSinCos, Scaler, Shuffle
from nosalro.rl.utils import train_sb3, cli
import pyfastsim as fastsim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = Scaler()
target_scaler = Scaler(_type='standard')
shuffle = Shuffle(seed=42)

transforms = Compose([
    shuffle,
    AngleToSinCos(angle_column=2),
    scaler.fit,
    scaler
])
dataset = StatesDataset(path='data/go_explore_1000.dat', transforms=transforms)

# In case output dims is different from input dims.
target_transforms = Compose([
    shuffle,
    target_scaler.fit,
    target_scaler
])
target_dataset = StatesDataset(path='data/go_explore_1000.dat', transforms=target_transforms)

vae = VariationalAutoencoder(input_dims=4, latent_dims=3, output_dims=3, hidden_sizes=[64,64], scaler=scaler).to(device)
vae = train(
    model = vae,
    epochs = 3000,
    lr = 1e-04,
    dataset = dataset,
    device = device,
    beta = 10,
    file_name = 'models/vae_models/dots_vae_cos_sin.pt',
    overwrite = False,
    weight_decay = 0,
    batch_size = 1024,
    target_dataset = target_dataset
)
# i, x_hat_var, mu, logvar = vae(torch.tensor(dataset[:]).to(device), device, True, scale=False)
# visualize([dataset[:], i.detach().cpu().numpy()], projection='2d')

# Env Init.
world_map = fastsim.Map('worlds/dots.pbm', 600)
robot = fastsim.Robot(10, fastsim.Posture(500., 500., 0.))
dataset.inverse([scaler])

action_space = Box(low=-1., high=1., shape=(2,), dtype=np.float32)
observation_space = Box(
    low=np.array([0, 0, -1, -1, -np.inf, -np.inf, -np.inf]),
    high=np.array([1, 1, 1, 1, np.inf, np.inf, np.inf]),
    shape=(7,),
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
    goals=dataset[np.random.randint(0, len(dataset[:]), size=1)],
    goal_conditioned_policy=True,
    latent_rep=True,
    observation_space=observation_space,
    action_space=action_space,
    random_start=False,
    max_steps=100,
    vae=vae,
    scaler=target_scaler,
    controller=DVController(),
    sigma_sq=100
)

actor_net = Actor(observation_space.shape[0], action_space.shape[0], 1)
critic_net = Critic(observation_space.shape[0], action_space.shape[0])

train_td3(env, actor_net, critic_net)

# eval_policy('models/policies/td3_test')