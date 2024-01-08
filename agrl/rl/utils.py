#!/usr/bin/env python
# encoding: utf-8
#|
#|    Copyright (c) 2022-2023 Computational Intelligence Lab, University of Patras, Greece
#|    Copyright (c) 2022-2023 Constantinos Tsakonas
#|    Copyright (c) 2022-2023 Konstantinos Chatzilygeroudis
#|    Authors:  Constantinos Tsakonas
#|              Konstantinos Chatzilygeroudis
#|    email:    tsakonas.constantinos@gmail.com
#|              costashatz@gmail.com
#|    website:  https://nosalro.github.io/
#|              http://cilab.math.upatras.gr/
#|
#|    This file is part of AGRL.
#|
#|    All rights reserved.
#|
#|    Redistribution and use in source and binary forms, with or without
#|    modification, are permitted provided that the following conditions are met:
#|
#|    1. Redistributions of source code must retain the above copyright notice, this
#|       list of conditions and the following disclaimer.
#|
#|    2. Redistributions in binary form must reproduce the above copyright notice,
#|       this list of conditions and the following disclaimer in the documentation
#|       and/or other materials provided with the distribution.
#|
#|    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#|    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#|    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#|    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#|    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#|    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#|    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#|    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#|    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#|    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#|
import copy
import pprint
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

def eval_policy(policy, env, eval_data, seed=42, graphics=False):
    eval_env = copy.deepcopy(env)
    eval_env.goals = eval_data
    eval_env.eval()
    if not hasattr(eval_env, 'discrete'):
        eval_env.discrete = False
    observations = eval_env.reset(seed=seed)
    cumulative_reward = 0
    final_state_error = []
    reward_acc = []
    if graphics:
        eval_env.render()
    time.sleep(10)
    while eval_env.eval_data_ptr < (len(eval_data)+1):
        action = policy.select_action(np.array(observations))
        observations, reward, done, _ = eval_env.step(action)
        cumulative_reward += reward
        if done:
            final_state_error.append(reward)
            reward_acc.append(cumulative_reward)
            if graphics:
                print("Final state error: ", final_state_error[-1], "Cummulative Reward: ", cumulative_reward)
            cumulative_reward = 0
            observation, done = eval_env.reset(seed=seed), False
    print("\n---------------------------------------")
    print(f"Evaluation over {len(eval_data)} episodes. \
        \nAverage reward overall: {np.mean(reward_acc):.3f} \
        \nAverage final state error: {np.mean(final_state_error):.3f}")
    # [print(f'{d:.3f}') for d in final_distance]
    print("---------------------------------------")
    if graphics:
        eval_env.close()
    return np.asarray(reward_acc), np.array(final_state_error)
