import gym

class Box(gym.spaces.Box):
    """
    Reasoning behind this Class extention is to scale the action output to specific range. For example
    if policy has tanh as activation function in the output layer, but the output needs to be converted
    to [0, 200] range, the scale function converts the action from [-1, 1] to [0, 200] range.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def scale(self, x, left_lim, right_lim):
        _high = self.high[:x.shape[-1]]
        _low = self.low[:x.shape[-1]]
        return (_high*(x + right_lim) - _low*(x + _high))/(right_lim - left_lim)