# import gymnasium as gym
import gym
import numpy as np

class Box(gym.spaces.Box):
    """
    Reasoning behind this Class extention is to scale the action output to specific range. For example
    if policy has tanh as activation function in the output layer, but the output needs to be converted
    to [0, 200] range, the scale function converts the action from [-1, 1] to [0, 200] range.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def unscale(self, x, left_lim, right_lim):
        _max = self.high[:x.shape[-1]]
        print(_max)
        _min = self.low[:x.shape[-1]]
        return (_max*(x + right_lim) - _min*(x + _max))/(right_lim - left_lim)

    def scale(self, x, left_lim, right_lim):
        _max = self.high[:x.shape[-1]]
        _min = self.low[:x.shape[-1]]
        return (right_lim - left_lim)*((x - _min) / (_max - _min)) + left_lim

    def clip(self, x):
        return np.clip(x, a_min=self.low, a_max=self.high)