import torch
import numpy as np
from agrl.env import KheperaDVControllerEnv, Box
from agrl.controllers import DVController
from agrl.rl.td3 import Actor, Critic, train_td3
from agrl.vae import StatesDataset, VariationalAutoencoder, train, visualize
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import pyfastsim as fastsim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = Scaler()
shuffle = Shuffle(seed=None)
transforms = Compose([shuffle, scaler.fit, scaler])
dataset = StatesDataset(np.loadtxt('data/maze_data_ge.dat')[:,:2], transforms=transforms)
vae = VariationalAutoencoder(input_dims=2, latent_dims=1, output_dims=2, hidden_sizes=[256,128, 64], scaler=scaler).to(device)
vae = train(
    model = vae,
    epochs = 5000,
    lr = 3e-04,
    dataset = dataset,
    device = device,
    beta = 2,
    file_name = 'models/vae_models/vae_maze_ge_1d_mse.pt',
    overwrite = True,
    weight_decay = 0,
    batch_size = 256,
    reconstruction_type = 'mse'
)
x_hat, x_hat_var, mu, logvar = vae(torch.tensor(dataset[:]).to(device), device, True, scale=False)
visualize([scaler(dataset[:], undo=True), scaler(x_hat.detach().cpu().numpy(), undo=True)], projection='2d', file_name='.tmp/images/maze_mse_1d_ge')

# Env Init.
world_map = fastsim.Map('worlds/maze_like.pbm', 600)
robot = fastsim.Robot(10, fastsim.Posture(510., 100., 0.))
dataset.inverse([scaler])

action_space = Box(low=-1., high=1., shape=(2,), dtype=np.float32)
observation_space = Box(
    low=np.array([0, 0, -np.pi, -np.pi, -np.inf]),
    high=np.array([1, 1, np.pi, np.pi,  np.inf]),
    shape=(5,),
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
    reward_type='mse',
    n_obs=4,
    goals=dataset[:],
    goal_conditioned_policy=True,
    latent_rep=True,
    observation_space=observation_space,
    action_space=action_space,
    random_start=False,
    max_steps=100,
    vae=vae,
    scaler=scaler,
    controller=DVController(),
    sigma_sq=50
)

actor_net = Actor(observation_space.shape[0], action_space.shape[0], 1)
critic_net = Critic(observation_space.shape[0], action_space.shape[0])
train_td3(env, actor_net, critic_net, np.loadtxt('data/eval_data/maze.dat'))