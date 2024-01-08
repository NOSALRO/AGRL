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
import numpy as np
from scipy.spatial.transform import Rotation as R
from .._base import BaseEnv


class AnymalEnv(BaseEnv):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_robot = copy.deepcopy(self.robot)
        self._dired_traj = self.init_robot.get_pos.T[0]
        self.name = 'AnymalEnv'
        self.prev_r = 0

    def _robot_act(self, action):
        action = self.action_space.clip(action)
        action = action.reshape(4,3,1)
        act = []
        for i in action:
            act.append(i)
        self.robot.integrate(act)

    def _reset_op(self):
        print(f'\t x: {np.linalg.norm(self._dired_traj[0] - self.robot.get_pos[0]):.02f}, y: {np.linalg.norm(self._dired_traj[1] - self.robot.get_pos[1]):.02f}, step: {self.iterations}', end='\r')
        self.prev_r = 0
        self.robot = copy.deepcopy(self.init_robot)

    def _truncation_fn(self):
        # return self.max_steps == self.iterations or self.robot.valid()
        return not self.robot.valid()

    def _observations(self):
        pos = self.robot.get_pos.T[0]
        orientation = R.from_matrix(self.robot.get_orientation).as_euler('zxy')
        vel = self.robot._base_vel.T[0]
        T_norm = np.divide(np.mod(self.robot._feet_phases, self.robot._T), self.robot._T_swing)
        return np.array([*pos, *orientation, *vel, *T_norm], dtype=np.float32)

    def _state(self):
        return None

    def _reward_fn(self, *args):
        observations, actions = args[0], args[1]
        phases = observations[-4:]
        x_dist = np.linalg.norm(self._dired_traj[0] - observations[0])
        y_dist = np.linalg.norm(self._dired_traj[1] - observations[1])
        self.prev_r = (0.5 * self.prev_r) + np.linalg.norm(phases) + 0.8 * np.log(1 + y_dist) - 0.2 * np.log(1 + x_dist)
        return  self.prev_r

    def _termination_fn(self, *args):
        return not self.robot.valid()
