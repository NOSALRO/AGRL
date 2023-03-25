import functools
import numpy as np
import torch

class Compose:

    def __init__(self, transforms):
        assert isinstance(transforms, (list, tuple)), "transforms must be a list or tuple."
        self.ignore_list = ['fit']
        if isinstance(transforms, tuple):
            transforms = list(transforms)
        self.transforms = transforms

    def __call__(self, x):
        if self.transforms:
            for _transform in self.transforms:
                x = _transform(x)
        return x

    def inverse(self, x, inverse_ops = []):

        if self.transforms and len(inverse_ops) == 0:
            for _transform in reversed(self.transforms):
                try:
                    if _transform.__name__ not in self.ignore_list:
                        x = _transform.inverse(x)
                except AttributeError:
                    x = _transform.inverse(x)
        else:
            for _transform in inverse_ops:
                try:
                    if _transform.__name__ not in self.ignore_list:
                        x = _transform.inverse(x)
                except AttributeError:
                    x = _transform.inverse(x)
        return x