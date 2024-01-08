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
from agrl.vae import StatesDataset, VariationalAutoencoder, Scaler, train, visualize
from agrl.rl import learn

dataset = StatesDataset(path="data/random_points.dat")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = Scaler('standard')
scaler.fit(dataset.get_data())
dataset.scale_data(scaler)
vae_arch = VariationalAutoencoder(2, 2, scaler=scaler, hidden_sizes=[128, 64]).to(device)
epochs = 500
lr = 3e-4
vae = train(
    vae_arch,
    epochs,
    lr,
    dataset,
    device,
    beta = 1,
    file_name='models/vae_models/vae_random_points_pdp.pt',
    overwrite=False,
    weight_decay=0,
    batch_size = 128,
)
# visualize(dataset.get_data(), projection='2d')
# visualize(vae(torch.tensor(dataset.get_data()), 'cpu', True, False)[0].detach().cpu().numpy(), projection='2d')

robot = Point([10, 10])
action_space = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
observation_space = gym.spaces.Box(
    low=np.array([0, 0, -np.inf, -np.inf]),
    high=np.array([600, 600, np.inf, np.inf]),
    shape=(4,),
    dtype=np.float32
)

env = SimpleEnv(
    robot=robot,
    reward_type='edl',
    target=None,
    n_obs=2,
    goals=dataset.get_data(),
    goal_conditioned_policy=True,
    latent_rep=True,
    observation_space=observation_space,
    action_space=action_space,
    random_start=True,
    max_steps=0,
    vae=vae,
    scaler=scaler
)
learn(env, device)