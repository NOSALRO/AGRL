import copy
import torch
import numpy as np
import pyfastsim as fastsim
from ._base import BaseEnv
from ..controllers import Controller


class KheperaWithControllerEnv(BaseEnv):

    def __init__(
        self,
        *,
        map,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.map = copy.deepcopy(map)
        self.initial_state = self._state()
        self.enable_graphics = False
        self.tmp_target = None
        self.low_level_controller = Controller(None)

    def _observations(self):
        _rpos = self.robot.get_pos()
        if self.goal_conditioned_policy:
            return np.array([_rpos.x(), _rpos.y(), np.cos(_rpos.theta()), np.sin(_rpos.theta()), *torch.tensor(self.condition).cpu().detach().numpy()])
        else:
            return np.array([_rpos.x(), _rpos.y(), np.cos(_rpos.theta()), np.sin(_rpos.theta())], dtype=np.float32)

    def render(self):
        if not hasattr(self, 'disp'):
            self.disp = fastsim.Display(self.map, self.robot)
            self.graphics = True
        self.map.clear_goals()
        self.map.add_goal(fastsim.Goal(*self.target[:2], 10, 1))
        if self.tmp_target is not None:
            self.map.add_goal(fastsim.Goal(*self.tmp_target, 10, 2))
        self.disp.update()

    def close(self):
        del self.disp
        self.map.clear_goals()
        self.graphics = False

    def _set_target(self, target_pos):
        self.target = target_pos

    def _set_robot_state(self, state):
        self.robot.set_pos(fastsim.Posture(*state))
        self.robot.move(0, 0, self.map, False)

    def _robot_act(self, action):
        self.tmp_target = (550 - 50) * action + 50
        for _ in range(1000):
            self.low_level_controller.set_target(self.tmp_target)
            cmds = self._controller()
            self.robot.move(*cmds, self.map, False)
            if self.enable_graphics:
                self.disp.update()

    def _controller(self):
        return self.low_level_controller.update(self._state())

    def _state(self):
        _rpos = self.robot.get_pos()
        return [_rpos.x(), _rpos.y(), _rpos.theta()]

    def _reward_fn(self, observation):
        if self.reward_type == 'mse':
            reward = np.linalg.norm(observation[:2] - self.target[:2], ord=2)
            return -reward
        elif self.reward_type == 'edl':
            scaled_obs = torch.tensor(self.scaler(observation)) if self.scaler is not None else observation
            reward = self.dist.log_prob(scaled_obs[:self.n_obs]).cpu().item()
            return reward
