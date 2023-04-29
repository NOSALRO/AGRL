import os
from nosalro.env.iiwa import IiwaEnv
from nosalro.env import Box
import sys
import time
import pickle
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO
from nosalro.rl.utils import eval_policy
import torch
import numpy as np
import RobotDART as rd
from nosalro.env.iiwa import IiwaEnv
from nosalro.env import Box
from nosalro.rl.td3 import Actor, Critic, train_td3
from nosalro.vae import StatesDataset, VariationalAutoencoder, train, visualize
from nosalro.transforms import Compose, AngleToSinCos, Scaler, Shuffle

def load_eval_data(f):
    sb3_algos = ['ppo', 'sac']

    with open(f'{f}/metadata.json') as metadata_file:
        metadata = json.load(metadata_file)
    with open(f'{f}/env.pickle','rb') as env_file:
        env = pickle.load(env_file)
    algorithm, steps = metadata['algorithm'], metadata['steps']
    if algorithm.lower() == 'sac':
        model_load = SAC.load
    elif algorithm.lower() == 'ppo':
        model_load = PPO.load

    logs = []
    logs_unordered = os.listdir(f"{f}/logs/")
    for files in logs_unordered:
        if len(files.split('_')) == 3:
            logs.append(int(files.split('_')[1]))
    logs.sort(key=int)
    for i in range(1, len(logs)+1):
        log_name = f"policy_{logs[i-1]}_steps"
        logs[i-1] = log_name
    if os.path.exists(f"{f}/policy.pickle"):
        logs.append(f"policy")
    for i in range(1, len(logs)):
        eol = '\t'
        if i % 5 == 0:
            eol = '\n'
        print(f"[{i}] {logs[i]}", end=eol)
    print(end='\n')
    while True:
        try:
            select_policy = int(input("Select policy to run: "))
            if select_policy - 1 < 0:
                raise IndexError
            policy_path = f"{f}/logs/{logs[select_policy-1]}.pickle" if logs[select_policy-1] != 'policy' else f"{f}/policy.pickle"
            if algorithm.lower() == 'td3':
                with open(policy_path,'rb') as model_file:
                    model = pickle.load(model_file)
            elif algorithm.lower() in sb3_algos:
                model = model_load(f"{f}/logs/{logs[select_policy-1]}.zip", env=env, print_system_info=True)
            break
        except IndexError as ie:
            print("Selected policy is not in list!")
            print('Try again')
        except FileNotFoundError as fnf:
            print("File does not exist!")
            print('Try again')
        except ValueError as ve:
            print("Incorrect value!")
            print('Try again')
        except KeyboardInterrupt:
            exit(0)
    env.set_max_steps(steps)
    return env, model, algorithm.lower(), logs

if __name__ == '__main__':
    env, model, _, _ = load_eval_data(sys.argv[1])
    eval_policy(model, env, eval_data=np.loadtxt(sys.argv[2]), graphics=True)