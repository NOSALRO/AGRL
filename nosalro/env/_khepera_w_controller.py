import copy
import torch
import numpy as np
import pyfastsim as fastsim
from ._khepera import  KheperaEnv
from ..controllers import Controller


class KheperaControllerEnv(KheperaEnv):

    def __init__(
        self,
        *,
        controller,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.initial_state = self._state()
        self.tmp_target = None
        self.low_level_controller = copy.deepcopy(controller)


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
        self.robot.set_pos(fastsim.Posture(*state))

    def _robot_act(self, action):
        self._controller(action)

    def _controller(self, tmp_target):
        tmp_target = self.observation_space.unscale(np.array(tmp_target), -1, 1)
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
        return np.array([_rpos.x(), _rpos.y(), _rpos.theta()])

    def _reward_fn(self, observation):
        if self.reward_type == 'mse':
            reward = np.linalg.norm(observation[:2] - self.target[:2])
            return -reward
        elif self.reward_type == 'edl':
            scaled_obs = torch.tensor(self.scaler(observation[:self.n_obs])) if self.scaler is not None else observation
            reward = self.dist.log_prob(scaled_obs[:self.n_obs]).cpu().item()
            return reward