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
import torch
import pyfastsim as fastsim
from ._base import BaseEnv


class KheperaEnv(BaseEnv):

    def __init__(
        self,
        world_map,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.world_map = copy.deepcopy(world_map)
        self.initial_state = self._state()
        self.enable_graphics = False

    def _observations(self):
        _rpos = self._state()
        if self.n_obs == 4:
            _obs = [_rpos[0], _rpos[1], np.cos(_rpos[2]), np.sin(_rpos[2])]
        elif self.n_obs == 3:
            _obs = [_rpos[0], _rpos[1], _rpos[2]]
        if self.goal_conditioned_policy:
            if isinstance(self.condition, torch.Tensor):
                self.condition = self.condition.cpu().detach().numpy()
            return np.array([*_obs, *self.condition])
        else:
            return np.array(_obs, dtype=np.float32)


    def render(self):
        if not hasattr(self, 'disp'):
            self.disp = fastsim.Display(self.world_map, self.robot)
            self.graphics = True
        self.world_map.clear_goals()
        self.world_map.add_goal(fastsim.Goal(*self.target[:2], 10, 1))
        self.disp.update()

    def close(self):
        if hasattr(self, 'disp'):
            del self.disp
        self.world_map.clear_goals()
        self.graphics = False

    def _set_robot_state(self, state):
        self.robot.set_pos(fastsim.Posture(*state))

    def _robot_act(self, action):
        self.robot.move(action[0], action[1], self.world_map, False)
        if self.enable_graphics:
            self.disp.update()

    def _state(self):
        _pose = self.robot.get_pos()
        return np.array([_pose.x(), _pose.y(), _pose.theta()])

    def _reward_fn(self, *args):
        observation, action = args
        if self.reward_type == 'mse':
            reward = np.linalg.norm(observation[:2] - self.target[:2]).item()
            return -reward
        elif self.reward_type == 'edl':
            scaled_obs = torch.tensor(self.scaler(observation[:self.n_obs])) if self.scaler is not None else observation
            reward = self.dist.log_prob(scaled_obs[:self.n_obs]).cpu().item()
            return reward

    def _termination_fn(self, *args):
        return np.linalg.norm(args[0][:2] - self.target[:2]) < 1e-02

    def _truncation_fn(self):
        return self.max_steps == self.iterations