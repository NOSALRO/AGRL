import os
import sys
import argparse
from datetime import datetime
import numpy as np
import torch
from gym_env import KheperaEnv
import stable_baselines3
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise

sys.path.append('../vae/')
from vae_arch import VAE, VariationalDecoder, VariationalEncoder
from data_loading import ObstacleData

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", help="choose RL algorithm (PPO/SAC).")
parser.add_argument("-m", "--mode", default='train', help="choose between train and eval mode.")
parser.add_argument("-g", "--graphics", action='store_true', help="enable graphics during training.")
parser.add_argument("--map", default="three_wall", help="choose map. Default: three_wall")
parser.add_argument("--file-name", default=None, help="file name to save model. If eval mode is on, then name of the model to load.")
args = parser.parse_args()
algorithm = args.algorithm
mode = args.mode
graphics = args.graphics
map_name = args.map
file_name = args.file_name if args.file_name else algorithm.lower()


# Load dataset and VAE.
datapoints = ObstacleData("../data_collection/datasets/no_wall.dat").get_data()
vae = torch.load(f"../models/vae_{map_name}_cos_sin.pt")

# Create Environment.
env = KheperaEnv(
    min_max_scale=[vae.min, vae.max],
    max_steps=1000,
    posture=[100, 100, 0],
    map_path=f"../data_collection/worlds/{map_name}.pbm",
    goals=datapoints,
    vae=vae
)

# Set up graphics.
if graphics:
    env.render()

# Set up RL algorithm.
if algorithm.lower() == 'sac':
    policy_kwargs = dict(net_arch=dict(pi=[16, 16], qf=[32, 16]))
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=1e-4,
        buffer_size=1000000,
        learning_starts=100,
        batch_size=512,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=NormalActionNoise(0, 0.2),
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        ent_coef='auto',
        target_update_interval=1,
        target_entropy='auto',
        use_sde=False,
        sde_sample_freq=-1,
        use_sde_at_warmup=True,
        tensorboard_log=None,
        policy_kwargs=None,
        verbose=1,
        seed=None,
        device='auto',
    )
elif algorithm.lower() == 'ppo':
    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch={'pi': [32, 16], 'vf': [64, 32]})
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
        normalize_advantage=False,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=1,
        target_kl=None,
        policy_kwargs=None,
        verbose=1,
        seed=None,
        device='auto'
    )

# Set up mode.
if mode.lower() == 'train':
    try:
        model.learn(total_timesteps=5e+6)
    except KeyboardInterrupt:
        print("Training stopped!")

    try:
        model.save(f"models/{file_name}_{datetime.now().strftime('%d-%m-%Y-%H%M')}")
    except FileNotFoundError:
        os.mkdir('models')
        model.save(f"models/{file_name}_{datetime.now().strftime('%d-%m-%Y-%H%M')}")

elif mode.lower() == 'eval':
    if algorithm.lower() == 'sac':
        model = SAC.load(file_name)
    elif algorithm.lower() == 'ppo':
        model = PPO.load(file_name)

if not graphics:
    env.render()

observation = env.reset()
while True:
    try:
        env.close()
        action, _ = model.predict(observation, deterministic=True)
        observation, _, done, _ = env.step(action, eval=True)
        if done:
            print("Enviroment Solved!")
    except KeyboardInterrupt:
        print("Exit")
        break