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
import pickle
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
transforms = Compose([shuffle, scaler.fit, scaler])
# dataset = StatesDataset(path='data/iiwa.dat', transforms=transforms)
data = np.random.uniform(
    low = np.array([-2.0943951, -2.0943951, -2.96705973, -2.0943951, -2.96705973, -2.0943951, -3.05432619]),
    high = np.array([2.0943951,2.0943951, 2.96705973, 2.0943951, 2.96705973, 2.0943951, 3.05432619]),
    size=(2000,7))
dataset = StatesDataset(path='data/iiwa_uniform.dat', transforms=transforms)
print(dataset[:])
vae = VariationalAutoencoder(input_dims=7, latent_dims=1, output_dims=7, hidden_sizes=[128,128], scaler=scaler).to(device)
vae = train(
    model = vae,
    epochs = 500,
    lr = 3e-04,
    dataset = dataset,
    device = device,
    beta = 1,
    file_name = '.tmp/iiwa_vae.pt',
    overwrite = False,
    weight_decay = 0,
    batch_size = 256,
    reconstruction_type='gnll',
    lim_logvar=False
)
# x_hat, x_hat_var, mu, logvar = vae(torch.tensor(dataset[:]).to(device), device, True, scale=False)
# x_hat = scaler(x_hat.cpu().detach().numpy(), undo=True)
# fig, ax = plt.subplots(2,3)
# dataset.inverse([scaler])
# robot = rd.Iiwa()
# p = []
# pr = []
# for i,j in zip(dataset, x_hat):
#     robot.set_positions([*i])
#     p.append(robot.body_pose_vec("iiwa_link_ee")[3:])
#     robot.set_positions([*j])
#     pr.append(robot.body_pose_vec("iiwa_link_ee")[3:])
# p = np.array(p)
# pr = np.array(pr)
# visualize([pr], projection='3d', file_name='.tmp/plot_r')
# visualize([p], projection='3d', file_name='.tmp/plot_or')
# for i in range(2):
#     for j in range(i*3,i*3+3):
#         ax[i][j-3].set_ylim(-10, 10)
#         ax[i][j-3].plot(dataset[:,j].reshape(-1,1))
#         ax[i][j-3].plot(x_hat[:,j].reshape(-1,1))
# plt.show()
# plt.savefig('.tmp/plot_r.png')

action_space = Box(
    low=np.array([-1.48352986, -1.48352986, -1.74532925, -1.30899694, -2.26892803, -2.35619449, -2.35619449]),
    high=np.array([1.48352986, 1.48352986, 1.74532925, 1.30899694, 2.26892803, 2.35619449, 2.35619449]),
    shape=(7,),
    dtype=np.float32
)

observation_space = Box(
    low = np.array([-2.0943951, -2.0943951, -2.96705973, -2.0943951, -2.96705973, -2.0943951, -3.05432619, -np.inf]),
    high = np.array([2.0943951, 2.0943951, 2.96705973, 2.0943951, 2.96705973, 2.0943951, 3.05432619, np.inf]),
    shape=(8,),
    dtype=np.float32
)
env = IiwaEnv(
    dt = 0.01,
    reward_type = 'none',
    n_obs = 7,
    goals = dataset[:],
    action_space=action_space,
    observation_space=observation_space,
    random_start=False,
    goal_conditioned_policy=True,
    latent_rep=True,
    max_steps=200,
    vae=vae,
    scaler=scaler
)
actor_net = Actor(observation_space.shape[0], action_space.shape[0], 1)
critic_net = Critic(observation_space.shape[0], action_space.shape[0])
train_td3(env, actor_net, critic_net, np.loadtxt('data/iiwa_uniform.dat')[:100])
