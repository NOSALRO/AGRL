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
        action_weight = 0,
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

    def _set_robot_state(self, state):
        self.robot.set_pos(fastsim.Posture(*state))

    def _robot_act(self, action):
        self._controller(action)

    def _controller(self, action):
        cmds = self.low_level_controller.update(action)
        for _ in range(5):
            self.robot.move(*cmds, self.world_map, False)
            if self.graphics:
                self.disp.update()

    def _state(self):
        _rpos = self.robot.get_pos()
        return np.array([_rpos.x(), _rpos.y(), _rpos.theta()])

    def _reward_fn(self, *args):
        if not self.eval_mode:
            observation, action = args[0], args[1]
            act = np.linalg.norm(action)
            if self.reward_type == 'mse':
                dist = np.linalg.norm(np.multiply(observation[:2], 600) - self.target[:2])
                return np.exp(-dist/self.sigma_sq)
            elif self.reward_type == 'edl':
                if len(self.scaler.mean) == 2:
                    scaled_obs = torch.tensor(self.scaler([observation[0]*600, observation[1]*600])) if self.scaler is not None else observation
                elif len(self.scaler.mean) == 4:
                    scaled_obs = torch.tensor(self.scaler([observation[0]*600, observation[1]*600, observation[2], observation[3]])) if self.scaler is not None else observation
                elif len(self.scaler.mean) == 3:
                    scaled_obs = torch.tensor(self.scaler(self._state())) if self.scaler is not None else observation
                # return self.dist.log_prob(scaled_obs[:self.n_obs]).cpu().item()/1e+4
                return (np.exp(((self.dist.log_prob(scaled_obs[:self.n_obs]).cpu().item() - self.logprob_min)/(self.logprob_max - self.logprob_min)/self.sigma_sq)+1e-10))
        else:
            return self._eval_reward()

    def _eval_reward(self):
        return np.linalg.norm(self._state()[:2] - self.target)

    def _reset_op(self):
        if self.reward_type == 'edl':
            logprobs = self.dist.log_prob(torch.tensor(self.scaler(self.goals)))
            self.logprob_max = logprobs.max()
            self.logprob_min = logprobs.min()

    def _observations(self):
        _rpos = self._state()
        if self.n_obs == 2:
            _obs = [_rpos[0]/600, _rpos[1]/600]
        elif self.n_obs == 4:
            _obs = [_rpos[0]/600, _rpos[1]/600, np.cos(_rpos[2]), np.sin(_rpos[2])]
        elif self.n_obs == 3:
            _obs = [_rpos[0]/600, _rpos[1]/600, _rpos[2]]
        if self.goal_conditioned_policy:
            if isinstance(self.condition, torch.Tensor):
                self.condition = self.condition.cpu().detach().numpy()
            return np.array([*_obs, *self.condition])
        else:
            return np.array(_obs, dtype=np.float32)
