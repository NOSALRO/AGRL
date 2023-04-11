import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
from .._base import BaseEnv


class AnymalEnv(BaseEnv):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.init_robot = copy.deepcopy(self.robot)
        self._dired_traj = self.init_robot.get_pos.T[0]
        self.name = 'AnymalEnv'
        self.prev_r = 0

    def _robot_act(self, action):
        action = self.action_space.clip(action)
        action = action.reshape(4,3,1)
        act = []
        for i in action:
            act.append(i)
        self.robot.integrate(act)

    def _reset_op(self):
        print(f'\t x: {np.linalg.norm(self._dired_traj[0] - self.robot.get_pos[0]):.02f}, y: {np.linalg.norm(self._dired_traj[1] - self.robot.get_pos[1]):.02f}, step: {self.iterations}', end='\r')
        self.prev_r = 0
        self.robot = copy.deepcopy(self.init_robot)

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
        observations, actions = args[0], args[1]
        phases = observations[-4:]
        x_dist = np.linalg.norm(self._dired_traj[0] - observations[0])
        y_dist = np.linalg.norm(self._dired_traj[1] - observations[1])
        self.prev_r = (0.5 * self.prev_r) + np.linalg.norm(phases) + 0.8 * np.log(1 + y_dist) - 0.2 * np.log(1 + x_dist)
        return  self.prev_r

    def _termination_fn(self, *args):
        return not self.robot.valid()
