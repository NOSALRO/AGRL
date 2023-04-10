import copy
import torch
import numpy as np
import pyfastsim as fastsim
from ._khepera import KheperaEnv


class KheperaDVControllerEnv(KheperaEnv):

    def __init__(
        self,
        *,
        controller,
        sigma_sq = 100,
        action_weight = 0.001,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.name = 'KheperaDVController'
        self.low_level_controller = copy.deepcopy(controller)
        self.sigma_sq = sigma_sq
        self.action_weight = action_weight

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

    def _controller(self, action):
        # tmp_target = self.observation_space.unscale(np.array(tmp_target), -1, 1)
        for _ in range(5):
            cmds = self.low_level_controller.update(action)
            self.robot.move(*cmds, self.world_map, False)
            if self.graphics:
                self.disp.update()

    def _state(self):
        _rpos = self.robot.get_pos()
        return np.array([_rpos.x(), _rpos.y(), _rpos.theta()])

    def _reward_fn(self, *args):
        observation, action = args
        collision = int(self.robot.get_collision())
        self.collision_weight += .0005 * collision
        self.collision_weight *= collision
        if self.reward_type == 'distance':
            act = np.linalg.norm(action)
            dist = np.linalg.norm(observation[:2]*600 - self.target[:2])
            return np.exp(-dist/self.sigma_sq) - (self.action_weight * act) - (self.collision_weight * np.log(1+dist))
        elif self.reward_type == 'edl':
            if len(self.scaler.mean) == 4:
                scaled_obs = torch.tensor(self.scaler(observation[:self.n_obs]*600)) if self.scaler is not None else observation
            elif len(self.scaler.mean) == 3:
                scaled_obs = torch.tensor(self.scaler(self._state())) if self.scaler is not None else observation
            act = np.linalg.norm(action)
            reward = np.exp(self.dist.log_prob(scaled_obs[:self.n_obs]).cpu().item()/self.sigma_sq) - self.action_weight * act
            return reward

    def _observations(self):
        _rpos = self._state()
        if self.n_obs == 4:
            _obs = [_rpos[0]/600, _rpos[1]/600, np.cos(_rpos[2]), np.sin(_rpos[2])]
        elif self.n_obs == 3:
            _obs = [_rpos[0]/600, _rpos[1]/600, _rpos[2]]
        if self.goal_conditioned_policy:
            if isinstance(self.condition, torch.Tensor):
                self.condition = self.condition.cpu().detach().numpy()
            return np.array([*_obs, *self.condition])
        else:
            return np.array(_obs, dtype=np.float32)

    def _reset_op(self):
        self.collision_weight = 0