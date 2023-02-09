import numpy as np
import copy

class Robot:

    def __init__(self):
        self.position = np.zeros(3, dtype=np.float32)
        self.initial_position = np.zeros(3, dtype=np.float32)

    def set_position(self, position):
        self.position = position
        self.initial_position = copy.deepcopy(position)

    def get_position(self):
        return self.position

    def reset(self):
        self.position = self.initial_position

    def step(self, dp):
        self.position += dp

    def state(self):
        return self.position
