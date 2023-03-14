import numpy as np


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
