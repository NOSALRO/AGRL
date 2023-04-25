import copy
import argparse
import os
import sys
import time
import pickle
import json
import torch
import numpy as np


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-name", help="File name to save model. If eval mode is on, then name of the model to load.")
    parser.add_argument("--steps", default=1e+3, type=int, help="Number of env steps. Default: 1e+03")
    parser.add_argument("--episodes", default=1e+5, type=int, help="Number of episodes. Default: 1e+05")
    parser.add_argument("--start-episode", default=20, type=int, help='Time steps initial random policy is used')
    parser.add_argument("--eval-freq", default=1e+5, type=int, help='How often (time steps) we evaluate')
    parser.add_argument("--actor-lr", default=1e-3, type=float, help='Actor learning rate')
    parser.add_argument("--critic-lr", default=1e-3, type=float, help='Critic learning rate')
    parser.add_argument("--expl-noise", default=0.1, type=float, help='Std of Gaussian exploration noise')
    parser.add_argument("--batch-size", default=256, type=int, help='Batch size for both actor and critic')
    parser.add_argument("--discount", default=0.99, type=float, help='Discount factor')
    parser.add_argument("--tau", default=0.005, type=float, help='Target network update rate')
    parser.add_argument("--policy-noise", default=0.2, type=float, help='Noise added to target policy during critic update')
    parser.add_argument("--noise-clip", default=0.5, type=float, help='Range to clip target policy noise')
    parser.add_argument("--policy-freq", default=2, type=int, help='Frequency of delayed policy updates')
    parser.add_argument("--update-steps", default=2048, type=int, help='Number of update steps for on-policy algorithms.')
    parser.add_argument("--epochs", default=20, type=int, help='Number of epochs for policy update.')
    parser.add_argument("--scheduling-episode", default=5000, type=int, help="Interval for scheduling repeat.")
    parser.add_argument("--checkpoint-episodes", default=1e+3, type=int, help="After how many episodes a checkpoint is stored.")
    parser.add_argument("--seed", default=None, help="Enviroment seed.")
    parser.add_argument("-g", "--graphics", action='store_true', help="Enable graphics during training.")
    args = parser.parse_args()
    return args

def eval_policy(policy, env, seed, eval_data, graphics=False):
    eval_env = copy.deepcopy(env)
    eval_env.eval()
    eval_env.goals = eval_data
    avg_reward = 0.
    eval_episodes = 0
    success = []
    if graphics:
        eval_env.render()
    while True:
        observation, done = eval_env.reset(seed=0), False
        eval_episodes += 1
        done_steps = 0
        while not done:
            action = policy.select_action(np.array(observation))
            state, reward, done, _ = eval_env.step(action)
            if np.linalg.norm(np.multiply(state[:2], 600) - eval_env.target) <= 15:
                done_steps += 1
            else:
                done_steps = 0
            avg_reward += reward
        if done_steps >= 30:
            success.append(True)
        else:
            success.append(False)
        if eval_env.eval_data_ptr == len(eval_data):
            break

    # avg_reward /= eval_episodes
    success = np.array(success)
    success_rate = (len(success[success == True])/len(success)) * 100

    print("\n---------------------------------------")
    # print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print(f"Evaluation over {eval_episodes} episodes: {success_rate:.3f}%")
    print("---------------------------------------")
    if graphics:
        eval_env.close()
    return avg_reward

def scheduler(obj, attributes, rate):
    for idx, attribute in enumerate(attributes):
        _current_value = getattr(obj, attribute)
        if isinstance(rate, (float)):
            if rate < 1:
                _new_value = max(0.01, rate*_current_value)
            else:
                _new_value = rate*_current_value
            setattr(obj, attribute, _new_value)
        else:
            if rate[idx] < 1:
                _new_value = max(0.01, rate[idx]*_current_value)
            else:
                _new_value = rate[idx]*_current_value
            setattr(obj, attribute, _new_value)
