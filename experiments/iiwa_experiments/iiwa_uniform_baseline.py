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
import RobotDART as rd
from agrl.env.iiwa import IiwaEnv
from agrl.env import Box
from agrl.rl.td3 import Actor, Critic, train_td3
from agrl.vae import StatesDataset, VariationalAutoencoder, train, visualize
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = Scaler()
shuffle = Shuffle(seed=None)
# transforms = Compose([shuffle, scaler.fit, scaler])
transforms = Compose([shuffle])
dataset = StatesDataset(path='data/iiwa_uniform.dat', transforms=transforms)
vae = VariationalAutoencoder(input_dims=7, latent_dims=1, output_dims=7, hidden_sizes=[256, 128, 64], scaler=None).to(device)
vae = train(
    model = vae,
    epochs = 500,
    lr = 3e-04,
    dataset = dataset,
    device = device,
    beta = 0.01,
    file_name = 'models/vae_models/vae_iiwa_uniform.pt',
    overwrite = False,
    weight_decay = 0,
    batch_size = 256,
    reconstruction_type='mse',
)
x_hat, x_hat_var, mu, logvar = vae(torch.tensor(dataset[:]).to(device), device, deterministic=True, scale=False)
x_hat = x_hat.cpu().detach().numpy()
# dataset.inverse([scaler])

robot = rd.Iiwa()
original_pose = []
reconstructed_pose = []
for i,j in zip(dataset, x_hat):
    robot.set_positions([*i])
    original_pose.append(robot.body_pose_vec("iiwa_link_ee")[3:])
    robot.set_positions([*j])
    reconstructed_pose.append(robot.body_pose_vec("iiwa_link_ee")[3:])
# visualize([np.array(original_pose)], projection='3d')
# visualize([np.array(reconstructed_pose)], projection='3d')
# plt.show()

action_space = Box(
    low=np.array([-1.48352986, -1.48352986, -1.74532925, -1.30899694, -2.26892803, -2.35619449, -2.35619449]),
    high=np.array([1.48352986, 1.48352986, 1.74532925, 1.30899694, 2.26892803, 2.35619449, 2.35619449]),
    shape=(7,),
    dtype=np.float32
)

observation_space = Box(
    low = np.array([-2.0943951, -2.0943951, -2.96705973, -2.0943951, -2.96705973, -2.0943951, -3.05432619]*2),
    high = np.array([2.0943951, 2.0943951, 2.96705973, 2.0943951, 2.96705973, 2.0943951, 3.05432619]*2),
    shape=(14,),
    dtype=np.float32
)

env = IiwaEnv(
    dt = 0.01,
    reward_type = 'mse',
    n_obs = 7,
    goals = dataset[:],
    action_space=action_space,
    observation_space=observation_space,
    random_start=False,
    goal_conditioned_policy=True,
    latent_rep=False,
    max_steps=200,
    vae=vae,
    scaler=None
)

actor_net = Actor(observation_space.shape[0], action_space.shape[0], 1)
critic_net = Critic(observation_space.shape[0], action_space.shape[0])
train_td3(env, actor_net, critic_net, np.loadtxt('data/eval_data/iiwa.dat'))
