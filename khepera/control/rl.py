import os
import sys
import argparse
from datetime import datetime
import numpy as np
import torch
from gym_env import KheperaEnv
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise

sys.path.append('../vae/')
from vae_arch import VAE, VariationalDecoder, VariationalEncoder
from data_loading import ObstacleData

device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu")

STEPS = 1e+03
EPISODES = 1e+03

# Parse arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-a", "--algorithm", help="choose RL algorithm (PPO/SAC).")
parser.add_argument("-m", "--mode", default='train', help="choose between train, continue and eval mode.")
parser.add_argument("-g", "--graphics", action='store_true', help="enable graphics during training.")
parser.add_argument("--map", default="three_wall", help="choose map. Default: three_wall")
parser.add_argument("--file-name", default=None, help="file name to save model. If eval mode is on, then name of the model to load.")
parser.add_argument("--steps", default=STEPS, help="number of env steps. Default: 1e+03")
parser.add_argument("--episodes", default=EPISODES, help="number of episodes. Default: 1e+03")
args = parser.parse_args()
algorithm = args.algorithm
mode = args.mode
graphics = args.graphics
map_name = args.map
file_name = args.file_name if args.file_name else algorithm.lower()
STEPS = int(args.steps)
EPISODES = int(args.episodes)


# Load dataset and VAE.
datapoints = ObstacleData("../data_collection/datasets/no_wall.dat").get_data()
vae = torch.load(f"../models/vae_{map_name}_cos_sin.pt")

# Create Environment.
env = KheperaEnv(
    min_max_scale=[vae.min, vae.max],
    max_steps=STEPS,
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
        action_noise=NormalActionNoise(0, 0.05),
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
        device=device,
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
        device=device
    )

# Set up mode.
if mode.lower() == 'train':
    try:
        model = model.learn(total_timesteps=STEPS*EPISODES, progress_bar=True)
    except KeyboardInterrupt:
        print("Training stopped!")

    try:
        model.save(f"models/{file_name}_{datetime.now().strftime('%d-%m-%Y-%H%M')}")
    except FileNotFoundError:
        os.mkdir('models')
        model.save(f"models/{file_name}_{datetime.now().strftime('%d-%m-%Y-%H%M')}")

elif mode.lower() == 'eval' or mode.lower() == 'continue':
    if algorithm.lower() == 'sac':
        model = SAC.load(file_name, env=env)
    elif algorithm.lower() == 'ppo':
        model = PPO.load(file_name, env=env)

if mode.lower() == 'continue':
    try:
        model = model.learn(total_timesteps=STEPS*EPISODES, progress_bar=True)
    except KeyboardInterrupt:
        print("Training stopped!")
        model.save(file_name)

if not graphics:
    env.render()

observation = env.reset()
model.set_env(env)
while True:
    try:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, done, _ = env.step(action, eval=True)
        if done:
            print("Enviroment Solved!")
    except KeyboardInterrupt:
        print("Exit")
        break