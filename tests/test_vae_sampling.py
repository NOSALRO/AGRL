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
import pickle
import numpy as np
from agrl.env import KheperaDVControllerEnv, Box
from agrl.controllers import DVController
from agrl.rl.td3 import Actor, Critic, train_td3
from agrl.vae import StatesDataset, VariationalAutoencoder, train, visualize
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import pyfastsim as fastsim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = Scaler()
with open('.tmp/alley_mobile_edl_1/env.pickle', 'rb') as ef:
    env = pickle.load(ef)

vae = env.vae
scaler = env.scaler
dataset = env.goals
goals = []

x_hat, x_hat_var, mu, log_var = vae(torch.tensor(dataset[:]).to(device), device, True, scale=True)
print(log_var.exp().detach().cpu().numpy().mean())
# for x in dataset:
#     for _ in range(10):
#         i, x_hat_var, mu, log_var = vae(torch.tensor(x).to(device), device, True, scale=True)
#         _, out, _ = vae.sample(1, device, mu.cpu(), log_var.mul(0.5).exp().cpu())
#         goals.append(scaler(out.cpu().detach().squeeze().numpy(), undo=True))
# visualize([np.array(goals)], projection='2d', file_name='.tmp/image_ge')
# visualize([np.array(goals)], projection='2d', file_name='.tmp/image_uniform')
# world_map = fastsim.Map('worlds/alley_right.pbm', 600)
# disp = fastsim.Display(world_map, fastsim.Robot(10, fastsim.Posture(0, 0,0)))
# for i in goals:
#     world_map.add_goal(fastsim.Goal(*i[:2], 10, 1))
# while True:
#     disp.update()

# i, x_hat_var, mu, logvar = vae(torch.tensor(dataset[:]).to(device), device, True, scale=False)
# visualize([dataset[:], i.detach().cpu().numpy()], projection='2d', file_name='.tmp/image')
