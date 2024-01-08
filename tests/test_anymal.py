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
from scipy.spatial.transform import Rotation
from agrl.robots import Anymal
from agrl.env.anymal import AnymalEnv
from agrl.env import Box
from agrl.rl.td3 import Actor, Critic, train_td3, eval_policy


anymal = Anymal()
action_lim = 2*anymal._mass * anymal._g
action_space = Box(
    low=np.zeros(4*3*1,),
    high=np.multiply(np.ones(4*3*1,), action_lim),
    shape=(4*3*1, ),
    dtype=np.float32
)

observation_space = Box(
    low = np.array([-np.inf, -np.inf, 0, -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, 0, 0, 0, 0]),
    high = np.array([np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf, 1, 1, 1, 1]),
    shape=(13,),
    dtype=np.float32
)

env = AnymalEnv(
    robot = anymal,
    reward_type = 'none',
    n_obs = 12,
    goals = [100, 100, 0],
    action_space=action_space,
    observation_space=observation_space,
    random_start=False,
    max_steps=400
)

actor_net = Actor(observation_space.shape[0], np.prod(action_space.shape), action_lim)
critic_net = Critic(observation_space.shape[0], np.prod(action_space.shape))

train_td3(env, actor_net, critic_net)
