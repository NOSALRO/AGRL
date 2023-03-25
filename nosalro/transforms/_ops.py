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
