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
        world_map,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.world_map = copy.deepcopy(world_map)
        self.initial_state = self._state()
        self.tmp_target = None
        self.low_level_controller = copy.deepcopy(Controller(None, 0.5, 0.))

    def _observations(self):
        _rpos = self._state()
        if self.n_obs == 4:
            _obs = [_rpos[0], _rpos[1], np.cos(_rpos[2]), np.sin(_rpos[2])]
        elif self.n_obs == 3:
            _obs = [_rpos[0], _rpos[1], _rpos[2]]
        if self.goal_conditioned_policy:
            return np.array([*_obs, *torch.tensor(self.condition).cpu().detach().numpy()])
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
        if self.graphics:
            del self.disp
            self.graphics = False
            self.world_map.clear_goals()

    def _set_target(self, target_pos):
        self.target = target_pos

    def _set_robot_state(self, state):
        # self.robot.move(0, 0, self.world_map, False)
        self.robot.set_pos(fastsim.Posture(*state))

    def _robot_act(self, action):
        self._controller(action)

    def _controller(self, tmp_target):
        tmp_target = self.__scale_action(np.array(tmp_target), 575, 25, -1, 1)
        self.low_level_controller.set_target(tmp_target)
        self.world_map.add_goal(fastsim.Goal(*tmp_target, 10, 2))
        for _ in range(50):
            cmds = self.low_level_controller.update(self._state())
            if cmds is not None:
                self.robot.move(*cmds, self.world_map, False)
            if self.graphics:
                self.disp.update()

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

    def _reset_op(self):
        self.low_level_controller.reset()

    @staticmethod
    def __scale_action(x, x_max, x_min, left_lim, right_lim):
        return (x_max*(x + right_lim) - x_min*(x + left_lim))/(right_lim - left_lim)