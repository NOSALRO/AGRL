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


class Controller:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.__sum_rot_error = 0
        self.__sum_lin_error = 0

    def set_target(self, target):
        self._target = target

    def reset(self):
        self.__sum_rot_error = 0
        self.__sum_lin_error = 0

    def error(self, current):
        xy_diff = np.array(self._target) - np.array(current[:-1])
        rot_error = np.arctan2(xy_diff[1], xy_diff[0]) - current[-1]
        lin_error = np.linalg.norm(xy_diff)
        return np.r_[rot_error, lin_error]

    def update(self, current):
        rot_error, lin_error = self.error(current)
        self.__sum_rot_error += np.abs(rot_error)
        self.__sum_lin_error += lin_error

        _rot_cmd = self.Kp*np.abs(rot_error) + self.Ki*self.__sum_rot_error
        _rot_cmd = np.clip(_rot_cmd, a_min=0., a_max=0.3)
        _rot_cmd = [0, _rot_cmd]
        _rot_cmd = np.multiply(np.sign(rot_error), _rot_cmd)

        if np.abs(rot_error) < 1e-01:
            lin_err = self.error(current)[1]
            _cmd = self.Kp*lin_error + self.Ki*self.__sum_lin_error
            _cmd = np.clip(_cmd, a_min=0., a_max=0.5)
            _cmd = [_cmd, _cmd] + _rot_cmd
            return _cmd if [lin_error, np.abs(rot_error)] > [1e-02, 1e-02] else None
        return _rot_cmd

class DVController:

    def __init__(self): ...

    def update(self, current):
        theta, vel = current[0], current[1]
        theta = theta * np.pi
        _cmd = [theta + vel, -theta + vel]
        return _cmd
