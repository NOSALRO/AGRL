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
import copy
import cma
from ._utils import set_model_params, get_model_params

def env_run(params, env, net, *, device='cpu', graphics=False, mode='train'):
    _env = copy.deepcopy(env)
    _net = copy.deepcopy(net)
    if mode == 'train':
        set_model_params(_net, params, device=device)
    reward = 0
    obs = _env.reset()
    if graphics:
        _env.render()
    action = _net(torch.tensor(obs, device=device))
    done = False
    while not done:
        obs, r, done, _ = _env.step(action.cpu().detach().numpy())
        reward += r
        action = _net(torch.tensor(obs, device=device))
    return -reward

def cma_run(env, net):
    best_solution = cma.evolution_strategy.fmin(env_run, x0=get_model_params(net), sigma0=0.5, options={"CMA_elitist" : True, "maxfevals" : 40000, "verbose" : True}, args=(env, net))
    return best_solution

def cma_eval(env, net, best_solution):
    set_model_params(net, best_solution[0], device='cpu')
    torch.save(net, 'models/policies/new_cma_policy.pt')
    # env_run(None, env, net, device='cpu', graphics=True)
