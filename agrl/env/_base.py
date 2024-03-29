#!/usr/bin/env python
# encoding: utf-8
#|
#|    Copyright (c) 2022-2023 Computational Intelligence Lab, University of Patras, Greece
#|    Copyright (c) 2022-2023 Constantinos Tsakonas
#|    Copyright (c) 2022-2023 Konstantinos Chatzilygeroudis
#|    Authors:  Constantinos Tsakonas
#|              Konstantinos Chatzilygeroudis
#|    email:    tsakonas.constantinos@gmail.com
#|              costashatz@gmail.com
#|    website:  https://nosalro.github.io/
#|              http://cilab.math.upatras.gr/
#|
#|    This file is part of AGRL.
#|
#|    All rights reserved.
#|
#|    Redistribution and use in source and binary forms, with or without
#|    modification, are permitted provided that the following conditions are met:
#|
#|    1. Redistributions of source code must retain the above copyright notice, this
#|       list of conditions and the following disclaimer.
#|
#|    2. Redistributions in binary form must reproduce the above copyright notice,
#|       this list of conditions and the following disclaimer in the documentation
#|       and/or other materials provided with the distribution.
#|
#|    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#|    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#|    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#|    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#|    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#|    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#|    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#|    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#|    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#|    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#|
import torch
import numpy as np
import gymnasium as gym
import copy
from ._spaces import Box


class BaseEnv(gym.Env):

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
        discrete = False,
        scaler = None,
        vae = None,
    ):
        # TODO: specify columns' index coordinates to calculate MSE, e.g. (x, y, z). Workaround: overload reward function.
        super().__init__()
        try:
            self.robot = copy.deepcopy(robot)
        except TypeError as te:
            print("Robot cannot be pickled")
            self.robot = robot
        self.reward_type = reward_type.lower()
        self.n_obs = n_obs
        self.max_steps = max_steps
        self.goal_conditioned_policy = goal_conditioned_policy
        self.goals = np.array(goals)
        self.latent_rep = latent_rep
        self.graphics = False
        self.scaler = scaler
        self.name = 'BaseEnv'
        self.iterations = 0
        self.eval_mode = False
        self.discrete = discrete

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
        if isinstance(random_start, (gym.spaces.Box, Box)):
            self.random_start = random_start
        elif random_start:
            self.random_start = self.observation_space
        else:
            self.random_start = random_start
            self.initial_state = self._state()

    def reset(self, *, seed=None, options=None):
        # Reset state.
        super().reset(seed=seed)
        self.iterations = 0
        # TODO: Create init_state def instead of this.
        self.initial_state = self.random_start.sample() if self.random_start is not False else self.initial_state
        self._set_robot_state(self.initial_state)
        self._set_target()
        self._reset_op() # In case of extra reset operations.
        return self._observations()

    def step(self, action):
        self.iterations += 1
        self._robot_act(action)
        observation = self._observations()
        reward = self._reward_fn(observation, action)
        terminated = False
        truncated = False
        if self._truncation_fn():
            truncated = True
        if self._termination_fn(observation):
            terminated = True
        if self.graphics:
            self.render()
        # return observation, reward, terminated, truncated, {}
        return observation, reward, truncated, {}

    def set_max_steps(self, max_steps):
        self.max_steps = max_steps

    def eval(self):
        self.eval_mode = True
        self.eval_data_idx = np.arange(len(self.goals))
        np.random.shuffle(self.eval_data_idx)
        self.eval_data_ptr = 0

    def train(self):
        self.eval_mode = False

    def _set_target(self):
        # Change target in case of multiple targets.
        if not self.eval_mode:
            _target = self.goals[np.random.randint(low=0, high=len(self.goals), size=(1,)).item()]
        else:
            _target = self.goals[self.eval_data_idx[self.eval_data_ptr]]
            self.eval_data_ptr += 1
        if self.goal_conditioned_policy:
            self._goal_conditioned_policy(_target)
        else:
            self.target = _target

    def _goal_conditioned_policy(self, target):
        # Get goal representation and distribution q.
        with torch.no_grad():
            if self.goal_conditioned_policy or self.reward_type == 'edl':
                x_hat, x_hat_var, mu, log_var = self.vae(torch.tensor(target, device=self.device).float(), self.device, True, True)
                if not self.eval_mode:
                    if not self.discrete:
                        latent, x_hat, x_hat_var = self.vae.sample(1, self.device, mu.cpu(), log_var.mul(0.5).exp().cpu())
                        latent = latent.view(latent.size()[0],)
                    else:
                        latent = mu.squeeze()
                    self.target = self.scaler(x_hat.cpu().squeeze().detach().numpy(), undo=True) if self.scaler is not None else x_hat.cpu().squeeze().detach().numpy()
                else:
                    latent = mu if not self.discrete else mu.squeeze()
                    self.target = target
                self.condition = latent.detach() if self.latent_rep else self.target
                if isinstance(self.condition, torch.Tensor):
                    self.condition = self.condition.cpu().detach().numpy()
                if self.reward_type == 'edl':
                    x_hat = x_hat.squeeze()
                    x_hat_var = x_hat_var.squeeze()
                    self.dist = torch.distributions.MultivariateNormal(x_hat.cpu(), torch.diag(x_hat_var.exp().cpu()))

    def _truncation_fn(self): ...
    def render(self): ...
    def close(self): ...
    def _reward_fn(self, observation): ...
    def _termination_fn(self, *args): ...
    def _observations(self): ...
    def _set_robot_state(self, state): ...
    def _robot_act(self, action): ...
    def _state(self): ...
    def _reward_fn(self, *args): ...
    def _reset_op(self): ...
