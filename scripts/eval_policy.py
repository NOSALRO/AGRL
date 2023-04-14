import os
import sys
import time
import pickle
import json
import torch
import numpy as np
from stable_baselines3 import SAC, PPO

def eval_policy():
    sb3_algos = ['ppo', 'sac']
    folder_path = sys.argv[1]
    env, model, algorithm = _load_eval_data(folder_path)
    env.eval()
    try:
        eval_data = np.loadtxt(sys.argv[2])
        env.goals = eval_data
    except IndexError:
        print("Eval data not found. Using original goals.")
    observation = env.reset()
    env.render()
    while True:
        try:
            if algorithm in sb3_algos:
                action, _ = model.predict(observation, deterministic=True)
            elif algorithm == 'td3':
                action = model.select_action(observation)
            observation, _, done, _ = env.step(action)
            if done:
                time.sleep(3)
                print('Env Reseted')
                observation = env.reset()
        except KeyboardInterrupt:
            print("Exit")
            exit(0)

def _load_eval_data(f):
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
    try:
        if algorithm.lower() == 'td3':
            with open(f"{f}/policy.pickle",'rb') as model_file:
                model = pickle.load(model_file)
        elif algorithm.lower() in sb3_algos:
            model = model_load(f"{f}/policy.zip", env=env, print_system_info=True)
    except FileNotFoundError:
        logs = []
        logs_unordered = os.listdir(f"{f}/logs/")
        for files in logs_unordered:
            if len(files.split('_')) == 3:
                logs.append(int(files.split('_')[1]))
        logs.sort(key=int)
        for i in range(1, len(logs)+1):
            eol = '\t'
            if i % 5 == 0:
                eol = '\n'
            log_name = f"policy_{logs[i-1]}_steps"
            print(f"[{i}] {log_name}", end=eol)
            logs[i-1] = log_name
        print(end='\n')
        logs = tuple(logs)
        while True:
            try:
                select_policy = int(input("Select policy to run: "))
                if select_policy - 1 < 0:
                    raise IndexError
                if algorithm.lower() == 'td3':
                    with open(f"{f}/logs/{logs[select_policy-1]}.pickle",'rb') as model_file:
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
    return env, model, algorithm.lower()

if __name__ == '__main__':
    eval_policy()
