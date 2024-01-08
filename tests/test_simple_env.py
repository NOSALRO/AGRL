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
from agrl.env import SimpleEnv
from agrl.robots import Point
from agrl.vae import StatesDataset, VariationalAutoencoder, train, visualize
from agrl.transforms import Scaler, Shuffle, Compose

device = torch.device("mps" if torch.cuda.is_available() else "cpu")
robot = Point([10, 10])
shuffle = Shuffle(seed=42)
scaler = Scaler()

transforms = Compose([
    shuffle,
    scaler.fit,
    scaler
    ])

dataset = StatesDataset(path="data/random_points.dat", transforms=transforms)

vae = VariationalAutoencoder(2, 2, output_dims=2, scaler=scaler)
epochs = 100
lr = 5e-4
vae = train(
    vae,
    epochs,
    lr,
    dataset,
    'cpu',
    beta = 0.1,
    file_name='vae_pdp.pt',
    overwrite=False,
    weight_decay=0
)

# visualize(dataset.get_data(), projection='2d')
# visualize(vae(torch.tensor(dataset.get_data()), 'cpu', True, False)[0].detach().cpu().numpy(), projection='2d')
action_space = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
observation_space = gym.spaces.Box(
    low=np.array([0, 0, -np.inf, -np.inf]),
    high=np.array([600, 600, np.inf, np.inf]),
    shape=(4,),
    dtype=np.float32
)

env = SimpleEnv(
    robot=robot,
    reward_type='mse',
    n_obs=2,
    goals=[20,20],#dataset.get_data()[[8000, 200]],
    goal_conditioned_policy=True,
    latent_rep=False,
    observation_space=observation_space,
    action_space=action_space,
    random_start=False,
    max_steps=100,
    vae=vae,
    scaler=scaler,
)

env.reset()
for _ in range(10): print(env.step([10, 0]))