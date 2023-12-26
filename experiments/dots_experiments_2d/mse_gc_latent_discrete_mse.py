import torch
import numpy as np
from agrl.env import KheperaDVControllerEnv, Box
from agrl.controllers import DVController
from agrl.rl.td3 import Actor, Critic, train_td3
from agrl.vae import StatesDataset, VQVAE, train, visualize
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import pyfastsim as fastsim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = Scaler()
shuffle = Shuffle(seed=None)
transforms = Compose([shuffle, scaler.fit, scaler])
dataset = StatesDataset(np.loadtxt('data/go_explore_xy.dat')[:,:2], transforms=transforms)
vqvae = VQVAE(input_dims=2, latent_dims=8, codes=10, output_dims=2, beta=0.25, hidden_sizes=[64, 64], scaler=scaler).to(device)
vqvae = vqvae.train(
    loss_fn=vqvae.loss_fn,
    model=vqvae,
    epochs=700,
    lr=3e-4,
    dataset = dataset,
    device = device,
    overwrite=False,
    file_name = 'models/vae_models/vae_dots_ge_2d_mse_discrete.pt',
    batch_size = 256,
    reconstruction_type='mse'
)

x_hat, x_hat_var, one_hot, _ = vqvae(torch.tensor(dataset[:], device=device), device, False, False)
x_hat = x_hat.detach().cpu().numpy()
visualize([scaler(dataset[:], True), scaler(x_hat[:], True)], file_name='.tmp/images/dots_mse_discrete')

# Env Init.
world_map = fastsim.Map('worlds/dots.pbm', 600)
robot = fastsim.Robot(10, fastsim.Posture(500., 500., 0.))
dataset.inverse([scaler])

action_space = Box(low=-1., high=1., shape=(2,), dtype=np.float32)
observation_space = Box(
    low=np.hstack((np.array([0, 0, -np.pi, -np.pi]), np.zeros(10))),
    high=np.hstack((np.array([1, 1, np.pi, np.pi]), np.ones(10))),
    shape=(14,),
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
    discrete=True,
    vae=vqvae,
    scaler=scaler,
    controller=DVController(),
    sigma_sq=100
)

actor_net = Actor(observation_space.shape[0], action_space.shape[0], 1)
critic_net = Critic(observation_space.shape[0], action_space.shape[0])
train_td3(env, actor_net, critic_net, np.loadtxt('data/eval_data/dots.dat'))