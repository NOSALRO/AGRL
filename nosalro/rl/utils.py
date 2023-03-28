import argparse
import os
import sys
import time
import pickle
import json
import torch
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.ppo import MlpPolicy


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", help="choose RL algorithm (PPO/SAC).")
    parser.add_argument("-m", "--mode", default='train', help="choose between train, continue and eval mode.")
    parser.add_argument("-g", "--graphics", action='store_true', help="enable graphics during training.")
    parser.add_argument("--file-name", default=None, help="file name to save model. If eval mode is on, then name of the model to load.")
    parser.add_argument("--steps", default=1e+3, help="number of env steps. Default: 1e+03")
    parser.add_argument("--episodes", default=1e+5, help="number of episodes. Default: 1e+05")
    args = parser.parse_args()
    algorithm = args.algorithm
    mode = args.mode
    graphics = args.graphics
    file_name = args.file_name if args.file_name else algorithm.lower()
    file_name = '.'.join(file_name.split('.')[:-1]) if file_name.split('.')[-1] == 'zip' else file_name
    steps = int(args.steps)
    episodes = int(args.episodes)
    return dict(algorithm=algorithm, mode=mode, graphics=graphics, file_name=file_name, steps=steps, episodes=episodes)

def train_sb3(env, device, algorithm, mode, graphics, file_name, steps, episodes):
    env.set_max_steps(steps)
    env.reset()
    metadata = {
        'algorithm': algorithm,
        'steps': steps,
    }
    if graphics:
        env.render()
    # Set up RL algorithm.
    if algorithm.lower() == 'sac':
        policy_kwargs = dict(net_arch=dict(pi = [32, 32], qf = [32,32]))
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=1e-4,
            buffer_size=1000000,
            learning_starts=1000,
            batch_size=1024,
            tau=0.005,
            gamma=0.99,
            train_freq=(2, 'episode'),
            gradient_steps=100,
            action_noise=NormalActionNoise(0, 0.05),
            replay_buffer_class=None,
            replay_buffer_kwargs=None,
            optimize_memory_usage=False,
            ent_coef='auto',
            target_update_interval=1,
            target_entropy='auto',
            use_sde=False,
            sde_sample_freq=-1,
            tensorboard_log=None,
            use_sde_at_warmup=True,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=None,
            device=device,
        )
    elif algorithm.lower() == 'ppo':
        policy_kwargs = dict(squash_output = True, net_arch=dict(pi = [32, 32], vf = [32,32]))
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=512,
            n_epochs=80,
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
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=None,
            device=device
        )

    # Set up mode.
    if mode.lower() == 'train':
        try:
            if not os.path.exists(file_name):
                os.mkdir(file_name)
            checkpoint_callback = CheckpointCallback(
            save_freq=1e+03,
            save_path=f"{file_name}/logs/",
            name_prefix='policy',
            save_replay_buffer=True,
            save_vecnormalize=True,
            verbose=True
            )
            model.learn(total_timesteps=steps*episodes, callback=checkpoint_callback, progress_bar=True)
        except KeyboardInterrupt:
            print("Training stopped!")

        model.save(f"{file_name}/policy")
        print(f"Saving model at {file_name}/{file_name.split('/')[-1]}.zip")
        with open(f"{file_name}/env.pkl", 'wb') as env_file:
            pickle.dump(env, env_file)
        with open(f"{file_name}/metadata.json", 'w') as metadata_file:
            json.dump(metadata, metadata_file)

def eval_policy():
    folder_path = sys.argv[1]
    with open(f'{folder_path}/metadata.json') as metadata_file:
        metadata = json.load(metadata_file)
    with open(f'{folder_path}/env.pkl','rb') as env_file:
        env = pickle.load(env_file)

    algorithm, steps = metadata['algorithm'], metadata['steps']
    env.set_max_steps(steps)
    env.reset()

    if algorithm.lower() == 'sac':
        model = SAC.load(f"{folder_path}/policy.zip", env=env)
    elif algorithm.lower() == 'ppo':
        model = PPO.load(f"{folder_path}/policy.zip", env=env)

    observation = env.reset()
    env.render()
    while True:
        try:
            action, _ = model.predict(observation, deterministic=True)
            observation, _, done, _ = env.step(action)
            if done:
                time.sleep(3)
                print('Env Reseted')
                env.reset()
        except KeyboardInterrupt:
            print("Exit")
            exit(0)
