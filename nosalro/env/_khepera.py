import copy
import numpy as np
import torch
import pyfastsim as fastsim
from ._base import BaseEnv


class KheperaEnv(BaseEnv):

    def __init__(
        self,
        world_map,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.world_map = copy.deepcopy(world_map)
        self.initial_state = self._state()
        self.enable_graphics = False

    def _observations(self):
        _rpos = self._state()
        if self.n_obs == 4:
            _obs = [_rpos[0], _rpos[1], np.cos(_rpos[2]), np.sin(_rpos[2])]
        elif self.n_obs == 3:
            _obs = [_rpos[0], _rpos[1], _rpos[2]]
        if self.goal_conditioned_policy:
            return np.array([*_obs, *self.condition])
        else:
            return np.array(_obs, dtype=np.float32)


    def render(self):
        if not hasattr(self, 'disp'):
            self.disp = fastsim.Display(self.world_map, self.robot)
            self.graphics = True
        self.world_map.clear_goals()
        self.world_map.add_goal(fastsim.Goal(*self.target[:2], 10, 1))
        self.disp.update()

    def close(self):
        if hasattr(self, 'disp'):
            del self.disp
        self.world_map.clear_goals()
        self.graphics = False

    def _set_robot_state(self, state):
        self.robot.set_pos(fastsim.Posture(*state))

    def _robot_act(self, action):
        self.robot.move(action[0], action[1], self.world_map, False)
        if self.enable_graphics:
            self.disp.update()

    def _state(self):
        _pose = self.robot.get_pos()
        return np.array([_pose.x(), _pose.y(), _pose.theta()])

    def _reward_fn(self, observation):
        if self.reward_type == 'mse':
            reward = np.linalg.norm(observation[:2] - self.target[:2]).item()
            return -reward
        elif self.reward_type == 'edl':
            scaled_obs = torch.tensor(self.scaler(observation[:self.n_obs])) if self.scaler is not None else observation
            reward = self.dist.log_prob(scaled_obs[:self.n_obs]).cpu().item()
            return reward
