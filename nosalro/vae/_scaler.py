import numpy as np


class Scaler:

    def __init__(self, _type = 'standard'):
        self._type = _type
        self.mean = None
        self.std = None
        self.min = None
        self.max = None

    def __call__(self, x):
        if self._type == 'standard':
            return (x - self.mean) / (self.std)
        elif self._type == 'min_max':
            return (x - self.min) / (self.max - self.min)

    def fit(self, x):
        if self._type == 'standard':
            self.mean = np.mean(x)
            self.std = np.std(x)
        elif self._type == 'min_max':
            self.min = np.min(x)
            self.max = np.max(x)
