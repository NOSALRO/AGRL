import os
import sys
import time
import pickle
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO
from nosalro.rl.utils import eval_policy

# def eval_policy(env=None, model=None, algorithm=None):
#     sb3_algos = ['ppo', 'sac']
#     folder_path = sys.argv[1]
#     if env is None and model is None and algorithm is None:
#         env, model, algorithm, _ = load_eval_data(folder_path)
#     env.eval()
#     env.reset()
#     env.render()
#     rewards, r = [], 0
#     episode = 0
#     try:
#         eval_data = np.loadtxt(sys.argv[-1])
#         env.goals = eval_data
#     except IndexError:
#         print("Eval data not found. Using original goals.")
#     observation = env.reset()
#     # while episode < 11:
#     while episode < len(eval_data)-2:
#         try:
#             if algorithm in sb3_algos:
#                 action, _ = model.predict(observation, deterministic=True)
#             elif algorithm == 'td3':
#                 action = model.select_action(observation)
#             observation, reward, done, _ = env.step(action)
#             r += -np.linalg.norm(env._state()[:2] - env.target[:2])
#             if done:
#                 print(r)
#                 # print('Env Reseted')
#                 rewards.append(r)
#                 r = 0
#                 observation = env.reset()
#                 episode += 1
#         except KeyboardInterrupt:
#             print("Exit")
#             exit(0)

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
    # try:
    #     if algorithm.lower() == 'td3':
    #         with open(f"{f}/policy.pickle",'rb') as model_file:
    #             model = pickle.load(model_file)
    #     elif algorithm.lower() in sb3_algos:
    #         model = model_load(f"{f}/policy.zip", env=env, print_system_info=True)
    # except FileNotFoundError:
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
