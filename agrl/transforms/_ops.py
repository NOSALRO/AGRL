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
from ._compose import Compose


class Shuffle:

    def __init__(self, seed=42):
        self.seed = seed
        np.random.seed(self.seed)
        self.shuffled_idx = None

    def __call__(self, x):
        if self.shuffled_idx is None:
            self.shuffled_idx = np.arange(len(x))
            np.random.shuffle(self.shuffled_idx)
        return x[self.shuffled_idx]

    def inverse(self, x):
        return x


class AngleToSinCos:

    def __init__(self, angle_column):
        self.angle_column = angle_column

    def __call__(self, x):
        _sines = np.sin(x[:, self.angle_column]).reshape(-1, 1)
        _cosines = np.cos(x[:,self.angle_column]).reshape(-1,1)
        x = np.concatenate((x, _cosines, _sines), axis=1)
        x = np.delete(x, self.angle_column, axis=1)
        return x

    def inverse(self, x):
        _angle = np.arccos(x[:, self.angle_column]).reshape(-1, 1)
        x = np.delete(x, [-2, -1], axis=1)
        x = np.concatenate((x, _angle), axis=1)
        return x


class Scaler:

    def __init__(self, _type = 'standard'):
        self._type = _type
        if self._type == 'standard':
            self.mean = None
            self.std = None
        elif self._type == 'min_max':
            self.min = None
            self.max = None

    def __call__(self, x, undo=False):
        if not undo:
            return self.__scale(x)
        elif undo:
            return self.__unscale(x)

    def inverse(self, x):
        return self.__call__(x, undo=True)

    def __scale(self, x):
        if self._type == 'standard':
            return np.array((x - self.mean) / (self.std), dtype=np.float32)
        elif self._type == 'min_max':
            return np.array((x - self.min) / (self.max - self.min), dtype=np.float32)

    def __unscale(self, x):
        if self._type == 'standard':
            return np.array(x*self.std + self.mean, dtype=np.float32)
        elif self._type == 'min_max':
            return np.array(x*(self.max - self.min) + self.min, np.float32)

    def fit(self, x):
        if self._type == 'standard':
            self.mean = np.mean(x, axis=0)
            self.std = np.std(x, axis=0)
        elif self._type == 'min_max':
            self.min = np.min(x, axis=0)
            self.max = np.max(x, axis=0)
        return x
