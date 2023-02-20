import sys
import argparse
import numpy as np
import torch
from gym_env import KheperaEnv
from utils import get_model_params
import stable_baselines3
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_checker import check_env

sys.path.append('../vae/')
from vae_arch import VAE, VariationalDecoder, VariationalEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", help="choose RL algorithm (PPO/SAC).")
parser.add_argument("-m", "--mode", default='train', help="choose between train and eval mode.")
parser.add_argument("-g", "--graphics", action='store_true', help="enable graphics during training.")
parser.add_argument("--map", default="three_wall", help="choose map. Default: three_wall")
args = parser.parse_args()
algorithm = args.algorithm
mode = args.mode
graphics = args.graphics
map_name = args.map


# Load dataset and VAE.
datapoints = np.loadtxt(f"../data_collection/datasets/{map_name}.dat")
vae = torch.load(f"../models/vae_{map_name}.pt")
target = datapoints[10000]
x_hat, x_hat_var, latent, _ = vae(torch.tensor(target, device=device).float(), device, True, True)
distribution = torch.distributions.MultivariateNormal(x_hat.cpu(), torch.diag(x_hat_var.mul(.5).exp().cpu()))

# Create Environment.
env = KheperaEnv(
    reward_dist=distribution,
    min_max_scale=[vae.min, vae.max],
    max_steps=800,
    posture=target-100,
    map_path=f"../data_collection/worlds/{map_name}.pbm"
)

# Set up graphics.
if graphics:
    env.render()
    env.set_target(target)

# Set up RL algorithm.
if algorithm.lower() == 'sac':
    policy_kwargs = dict(net_arch=dict(pi=[16, 16], qf=[32, 16]))
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=0.0003,
        verbose=1,
        batch_size=128,
        policy_kwargs=policy_kwargs,
        learning_starts=150,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=1
    )
elif algorithm.lower() == 'ppo':
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=2,
        target_kl=0.5,
        policy_kwargs=None,
        verbose=1,
        seed=None,
        device='auto'
    )

# Set up mode.
if mode.lower() == 'train':
    try:
        model.learn(total_timesteps=1e+4)
    except KeyboardInterrupt:
        print("Training stopped!")

    model.save(algorithm)
    env.render()
    env.set_target(target)
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)

elif mode.lower() == 'eval':
    if algorithm.lower() == 'sac':
        model = SAC.load('sac')
    elif algorithm.lower() == 'ppo':
        model = PPO.load('ppo')

    env.render()
    env.set_target(target)
    observation = env.reset()
    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, done, _ = env.step(action, eval=True)
        if done:
            print("Enviroment Solved!")