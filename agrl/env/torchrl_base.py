from collections import defaultdict
from typing import Optional
import torch
import numpy as np
import gymnasium as gym
import copy
import torchrl
from tensordict.nn import TensorDictModule
from tensordict.tensordict import TensorDict, TensorDictBase
from ._spaces import Box


class BaseEnv(torchrl.envs.EnvBase):

    def __init__(
        self,
        *,
        robot,
        reward_type,
        n_obs,
        max_steps,
        observation_space,
        action_space,
        goals,
        random_start = False,
        goal_conditioned_policy = False,
        latent_rep = False,
        scaler = None,
        vae = None,
    ):
        # TODO: specify columns' index coordinates to calculate MSE, e.g. (x, y, z). Workaround: overload reward function.
        super().__init__()
        self.robot = copy.deepcopy(robot)
        self.reward_type = reward_type.lower()
        self.n_obs = n_obs
        self.max_steps = max_steps
        self.goal_conditioned_policy = goal_conditioned_policy
        self.goals = np.array(goals)
        self.latent_rep = latent_rep
        self.graphics = False
        self.scaler = scaler

        assert isinstance(observation_space, (gym.spaces.Box, Box)), "observation_space type must be gym.spaces.Box"
        assert isinstance(action_space, (gym.spaces.Box, Box)), "action_space type must be gym.spaces.Box"
        self.action_space = action_space
        self.observation_space = observation_space

        if vae is not None:
            self.device = torch.device("cuda" if torch.has_cuda else "cpu")
            self.vae = vae.to(self.device)
        if len(self.goals.shape) == 1:
            self.goals = np.expand_dims(self.goals, 0)

        # Incase starting position should be bounded.
        if isinstance(random_start, gym.spaces.Box):
            self.random_start = random_start
        elif random_start:
            self.random_start = self.observation_space
        else:
            self.random_start = random_start
            self.initial_state = self._state()

    def _reset(self):
        # Reset state.
        self.iterations = 0
        self._reset_op() # In case of extra reset operations.
        self.initial_state = self.random_start.sample()[:self.n_obs] if self.random_start is not False else self.initial_state
        self._set_robot_state(self.initial_state)

        # Change target in case of multiple targets.
        _target = self.goals[np.random.randint(low=0, high=len(self.goals), size=(1,)).item()]
        self._set_target(_target)

        # Get goal representation and distribution q.
        if self.goal_conditioned_policy or self.reward_type == 'edl':
            x_hat, x_hat_var, latent, _ = self.vae(torch.tensor(self.target, device=self.device).float(), self.device, True, True)
            self.condition = latent if self.latent_rep else self.target
            if isinstance(self.condition, torch.tensor):
                self.condition = self.condition.cpu().detach().numpy()
            if self.reward_type == 'edl':
                # TODO: Check if reparam should be used here.
                # x_hat, x_hat_var, latent, _ = self.vae(torch.tensor(self.target, device=self.device).float(), self.device, False, True)
                self.dist = torch.distributions.MultivariateNormal(x_hat.cpu(), torch.diag(x_hat_var.exp().cpu()))
        return TensorDict(
            {
                'observation': self._observations(),
            },
            batch_size=self._observations().shape
        )

    def _step(self, action):
        self.iterations += 1
        self._robot_act(action)
        observation = self._observations()
        reward = self._reward_fn(observation)
        terminated = False
        truncated = False
        if self._truncation_fn():
            truncated = True
        if self._termination_fn(observation):
            terminated = True
        if self.graphics:
            self.render()
        return TensorDict({
            'next': {
                'observation': observation,
                'reward': reward,
                'done': truncated,
                'info': {}
            }
        })

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _reward_fn(self, observation):
        if self.reward_type == 'mse':
            reward = np.linalg.norm(observation[:self.n_obs] - self.target)
            return -reward
        elif self.reward_type == 'edl':
            scaled_obs = torch.tensor(self.scaler(observation[:self.n_obs])) if self.scaler is not None else observation
            reward = self.dist.log_prob(scaled_obs[:self.n_obs]).cpu().item()
            return reward

    def set_max_steps(self, max_steps):
        self.max_steps = max_steps

    def _set_target(self, goal):
        self.target = goal

    def _termination_fn(self, *args):
        return np.linalg.norm(args[0][:2] - self.target[:2]) < 1e-02

    def _truncation_fn(self):
        return self.max_steps == self.iterations

    def render(self): ...
    def close(self): ...
    def _observations(self): ...
    def _set_robot_state(self, *args): ...
    def _robot_act(self, action): ...
    def _state(self): ...
    def _reward_fn(self, observation): ...
    def _reset_op(self): ...