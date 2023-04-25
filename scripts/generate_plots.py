import os
import sys
import time
import pickle
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC, PPO
sys.path.append('./scripts/')
from eval_policy import *

def eval_logs(f):
    # folder_path = sys.argv[1]
    sb3_algos = ['ppo', 'sac']
    folder_path = f
    env, model, algorithm, logs = load_eval_data(folder_path)
    log_rewards = []
    steps = []
    for log in logs:
        print(log)
        steps.append(int(log.split('_')[1]))
        if algorithm.lower() == 'td3':
            with open(f"{folder_path}/logs/{log}.pickle",'rb') as model_file:
                model = pickle.load(model_file)
        elif algorithm.lower() in sb3_algos:
            model = model_load(f"{folder_path}/logs/{log}.zip", env=env, print_system_info=True)
        log_rewards.append(eval_policy(env, model, algorithm))
    median_log_rewards = np.mean(log_rewards,axis=1)
    plt.title(folder_path.split('/')[-1])
    plt.plot(np.array(steps), median_log_rewards)


if __name__ == '__main__':
    eval_logs(sys.argv[1])
    eval_logs(sys.argv[2])
    plt.gca().legend(('GE','Uniform'))
    plt.show()