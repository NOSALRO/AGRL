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
import numpy as np
import torch
import gym
from agrl.env import KheperaWithControllerEnv
from agrl.vae import StatesDataset, VariationalAutoencoder, train, visualize
from agrl.es import cma_run, env_run, cma_eval, PolicyNet
import pyfastsim as fastsim

net = PolicyNet(
        input_dim=4,
        output_dim=2,
        hidden_layers=[(16, torch.nn.ReLU()), (16, torch.nn.ReLU())],
        output_activation=torch.nn.Tanh(),
    )

dataset = StatesDataset(path="data/no_wall.dat")
world_map = fastsim.Map('worlds/no_wall.pbm', 600)
robot = fastsim.Robot(10, fastsim.Posture(50., 40., np.pi))
action_space = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
observation_space = gym.spaces.Box(
    low=np.array([0, 0, -1, -1]),
    high=np.array([600, 600, 1, 1]),
    shape=(4,),
    dtype=np.float32
)

start_space = gym.spaces.Box(
    low=np.array([50, 50, -np.pi]),
    high=np.array([550, 550, np.pi]),
    shape=(3,),
    dtype=np.float32
)

env = KheperaWithControllerEnv(
    robot=robot,
    world_map=world_map,
    reward_type='mse',
    target=None,
    n_obs=4,
    goals=dataset[[100]],
    goal_conditioned_policy=False,
    latent_rep=False,
    observation_space=observation_space,
    action_space=action_space,
    random_start=start_space,
    max_steps=60,
)

best_sol = cma_run(env, net)
cma_eval(env, net, best_sol)

## best_sol = cma_run(env, net)
#net = torch.load('models/policies/cma_policy.pt')
#cma_eval(env, net, None)