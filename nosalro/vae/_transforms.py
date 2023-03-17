import numpy as np
import torch

class Transform:

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple)), "transforms must be a list or tuple."
        if isinstance(transforms, tuple):
            transforms = list(transforms)
        self.transforms = transforms

    def __call__(self, x):
        if self.transforms:
            for _transform in self.transforms:
                x = _transform(x)
            return x

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


class AngleToSinCos:

    def __init__(self, angle_column):
        self.angle_column = angle_column

    def __call__(self, x):
        _sines = np.sin(x[:, self.angle_column]).reshape(-1, 1)
        _cosines = np.cos(x[:,self.angle_column]).reshape(-1,1)
        x = np.concatenate((x, _cosines, _sines), axis=1)
        x = np.delete(x, self.angle_column, axis=1)
        return x

class Scaler:

    def __init__(self, _type = 'standard'):
        self._type = _type
        self.applied = False
        if self._type == 'standard':
            self.mean = None
            self.std = None
        elif self._type == 'min_max':
            self.min = None
            self.max = None

    def __call__(self, x, undo=False):
        if not undo:
            self.applied = True
            return self.__scale(x)
        elif undo and self.applied:
            return self.__unscale(x)
        elif undo and not self.applied:
            print("Data is not scaled")

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
