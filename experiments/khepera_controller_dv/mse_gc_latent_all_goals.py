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
import torch
import numpy as np
from agrl.env import KheperaDVControllerEnv, Box
from agrl.controllers import DVController
from agrl.rl.td3 import Actor, Critic, train_td3, eval_policy
from agrl.vae import StatesDataset, VariationalAutoencoder, train, visualize
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
from agrl.rl.utils import train_sb3, cli
import pyfastsim as fastsim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = Scaler()
target_scaler = Scaler(_type='standard')
shuffle = Shuffle(seed=42)

transforms = Compose([
    shuffle,
    AngleToSinCos(angle_column=2),
    scaler.fit,
    scaler
])
dataset = StatesDataset(path='data/go_explore_1000.dat', transforms=transforms)

# In case output dims is different from input dims.
target_transforms = Compose([
    shuffle,
    target_scaler.fit,
    target_scaler
])
target_dataset = StatesDataset(path='data/go_explore_1000.dat', transforms=target_transforms)

vae = VariationalAutoencoder(input_dims=4, latent_dims=3, output_dims=3, hidden_sizes=[64,64], scaler=scaler).to(device)
vae = train(
    model = vae,
    epochs = 3000,
    lr = 1e-04,
    dataset = dataset,
    device = device,
    beta = 10,
    file_name = 'models/vae_models/dots_vae_cos_sin.pt',
    overwrite = False,
    weight_decay = 0,
    batch_size = 1024,
    target_dataset = target_dataset
)
# i, x_hat_var, mu, logvar = vae(torch.tensor(dataset[:]).to(device), device, True, scale=False)
# visualize([dataset[:], i.detach().cpu().numpy()], projection='2d')

# Env Init.
world_map = fastsim.Map('worlds/dots.pbm', 600)
robot = fastsim.Robot(10, fastsim.Posture(500., 500., 0.))
dataset.inverse([scaler])

action_space = Box(low=-1., high=1., shape=(2,), dtype=np.float32)
observation_space = Box(
    low=np.array([0, 0, -1, -1, -np.inf, -np.inf, -np.inf]),
    high=np.array([1, 1, 1, 1, np.inf, np.inf, np.inf]),
    shape=(7,),
    dtype=np.float32
)
start_space = Box(
    low=np.array([25, 25, -np.pi]),
    high=np.array([575, 575, np.pi]),
    shape=(3,),
    dtype=np.float32
)

env = KheperaDVControllerEnv(
    robot=robot,
    world_map=world_map,
    reward_type='distance',
    n_obs=4,
    goals=dataset[:],
    goal_conditioned_policy=True,
    latent_rep=True,
    observation_space=observation_space,
    action_space=action_space,
    random_start=False,
    max_steps=100,
    vae=vae,
    scaler=target_scaler,
    controller=DVController(),
    sigma_sq=100
)

actor_net = Actor(observation_space.shape[0], action_space.shape[0], 1)
critic_net = Critic(observation_space.shape[0], action_space.shape[0])

train_td3(env, actor_net, critic_net)

# eval_policy('models/policies/td3_test')