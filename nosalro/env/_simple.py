import torch
import numpy as np
from ._base import BaseEnv


class SimpleEnv(BaseEnv):

    def __init__(self, **kwargs):
        # TODO: specify columns' index coordinates to calculate MSE, e.g. (x, y, z)
        super().__init__(**kwargs)
        self.initial_state = self._state()

    def _observations(self):
        if self.goal_conditioned_policy:
            return np.array([*self.robot.get_position(), *torch.tensor(self.condition).cpu().detach().numpy()])
        else:
            return np.array(self.robot.get_position(), dtype=np.float32)

    def _set_target(self, target_pos):
        self.target = target_pos

    def _set_robot_state(self, state):
        self.robot.set_position(state)

    def _robot_act(self, action):
        self.robot.move(action)

    def _state(self):
        return self.robot.get_position()

    def render(self):
        print(f"Robot Pos: {self._state()} | Target Pos: {self.target}", end='\r')
        self.graphics = True

    def close(self):
        self.graphics = False
