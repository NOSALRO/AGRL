import numpy as np


class Point:

    def __init__(self, position=None):
        self.position = position

    def set_position(self, position):
        self.position = self._bound(position)

    def get_position(self):
        return self.position

    def move(self, dp):
        self.position = self._bound(self.position + dp)

    def _bound(self, p):
        pos = np.clip(p, a_min=0, a_max=600, dtype=np.float32)
        return pos