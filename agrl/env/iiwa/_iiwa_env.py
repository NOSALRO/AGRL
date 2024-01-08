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
import types
import copy
import torch
import numpy as np
import RobotDART as rd
import dartpy
from .._base import BaseEnv


class IiwaEnv(BaseEnv):

    def __init__(self, dt, **kwargs):
        self.dt = dt
        self.init_pos = [1., np.pi / 3., 0., -np.pi / 3., 0., np.pi / 3., 0.]
        self.init_custom_objects()
        super().__init__(robot=self.robot, **kwargs)
        self.name = 'IiwaEnv'
        self.episode = 0

    def render(self):
        self.graphics = True
        if not hasattr(self, 'graphics_component'):
            gconfig = rd.gui.GraphicsConfiguration(1920, 1080) # Create a window of 1024x768 resolution/size
            self.graphics_component = rd.gui.Graphics(gconfig) # create graphics object with configuration
            self.simu.set_graphics(self.graphics_component)
            self.graphics_component.look_at([0., 3., 2.], [0., 0., 0.])
            self.simu.step_world()

    def close(self):
        self.graphics = False
        if hasattr(self, 'graphic_component'):
            self.simu.set_graphics(None)
            del self.graphics_component
            self.simu.scheduler().set_sync(False)

    def _reset_op(self):
        self.episode += 1
        self.robot_ghost.set_positions([*self.target])

    def _robot_act(self, action):
        action = self.action_space.clip(action)
        action = action.reshape(self.action_space.shape[0],1)
        self.robot.set_commands(action)
        self.simu.step_world()

    def _truncation_fn(self):
        return True if self.iterations == self.max_steps else False

    def _set_robot_state(self, state):
        self.robot.set_positions(state)
        self.robot.set_commands(np.zeros((self.action_space.shape[0],1)))

    def _observations(self):
        _obs = self.robot.positions()
        if self.goal_conditioned_policy:
            if isinstance(self.condition, torch.Tensor):
                self.condition = self.condition.cpu().detach().numpy()
            return np.array([*_obs, *self.condition])
        else:
            return np.array(_obs, dtype=np.float32)

    def _state(self):
        return self.robot.positions()

    def _set_target(self):
        super()._set_target()
        self.robot_ghost.set_positions([*self.target])

    def _reward_fn(self, *args):
        obs, actions = args[0], args[1]
        if not self.eval:
            if self.reward_type == 'mse':
                return -np.linalg.norm(obs[:self.n_obs] - self.target) - (00.1 * np.linalg.norm(actions))
        else:
            return -np.linalg.norm(obs[:self.n_obs] - self.target)

    def init_custom_objects(self):
        self.robot = rd.Iiwa()
        self.robot.set_actuator_types("velocity")
        self.robot.set_positions(self.init_pos)
        self.robot_ghost=self.robot.clone_ghost()

        self.simu = rd.RobotDARTSimu(self.dt)
        self.simu.add_robot(self.robot)
        self.simu.add_checkerboard_floor()
        self.simu.add_robot(self.robot_ghost)

    def __getstate__(self):
        state = self.__dict__
        state['robot'] = None
        state['simu'] = None
        state['robot_ghost'] = None
        new_state = copy.deepcopy(state)
        self.init_custom_objects()
        return new_state

    def __setstate__(self, newstate):
        self.__dict__ = newstate
        self.init_custom_objects()

    def __deepcopy__(self, memo):
        del self.robot
        del self.simu
        del self.robot_ghost
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None
        cp = copy.deepcopy(self, memo)
        self.__deepcopy__ = deepcopy_method
        cp.__deepcopy__ = types.MethodType(deepcopy_method.__func__, cp)
        self.init_custom_objects()
        return cp

    def __copy__(self):
        del self.robot
        del self.simu
        del self.robot_ghost
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        self.init_custom_objects()
        return obj
