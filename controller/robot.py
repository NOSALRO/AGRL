import numpy as np
import copy

class Robot:

    def __init__(self):
        self.position = np.zeros(3, dtype=np.float32)
        self.initial_position = np.zeros(3, dtype=np.float32)
        #self.orientation = np.eye(3, dtype=np.float64)
        self.orientation = np.zeros(3, dtype=np.float64)
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)

    def set_position(self, position):
        self.position = position
        self.initial_position = copy.deepcopy(position)

    def get_position(self):
        return self.position

    def reset(self):
        self.position = self.initial_position

    def step(self, action):
        self.position += action[:3]
        self.velocity += action[3:6]
        self.orientation += action[6:9]
        self.angular_velocity += action[9:12]

    def state(self):
        return np.vstack((
            self.position,
            self.velocity,
            self.orientation,
            self.angular_velocity
        )).flatten()
