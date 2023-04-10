import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
from .._base import BaseEnv


class AnymalEnv(BaseEnv):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_robot = copy.deepcopy(self.robot)
        self.name = 'AnymalEnv'

    def _robot_act(self, action):
        self.robot.integrate(action)

    def _reset_op(self):
        self.robot = self.init_robot

    def _truncation_fn(self):
        # return self.max_steps == self.iterations or self.robot.valid()
        return not self.robot.valid()

    def _observations(self):
        pos = self.robot.get_pos.T[0]
        orientation = R.from_matrix(self.robot.get_orientation).as_euler('zxy')
        vel = self.robot._base_vel.T[0]
        T_norm = np.divide(np.mod(self.robot._feet_phases, self.robot._T), self.robot._T_swing)
        return np.array([*pos, *orientation, *vel, *T_norm], dtype=np.float32)

    def _state(self):
        return None

    def _reward_fn(self, *args):
        pass

    def _termination_fn(self, *args):
        return self.robot.valid()
