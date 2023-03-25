import torch
import gym
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
        goals = None,
        random_start = False,
        goal_conditioned_policy = False,
        latent_rep = True,
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
        self.goals = torch.tensor(goals)
        self.latent_rep = latent_rep
        self.graphics = False
        self.scaler = scaler

        assert isinstance(observation_space, (gym.spaces.Box, Box)), "observation_space type must be gym.spaces.Box"
        assert isinstance(action_space, (gym.spaces.Box, Box)), "action_space type must be gym.spaces.Box"
        self.action_space = action_space
        self.observation_space = observation_space
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

        if vae is not None:
            self.vae = vae.to(self.device)
        if len(self.goals.size()) == 1:
            self.goals = torch.unsqueeze(self.goals, 0)

        # Incase starting position should be bounded.
        if isinstance(random_start, gym.spaces.Box):
            self.random_start = random_start
        elif random_start:
            self.random_start = self.observation_space
        else:
            self.random_start = random_start
            self.initial_state = self._state()

    def reset(self):
        # Reset state.
        self.iterations = 0
        self._reset_op() # In case of extra reset operations.
        self.initial_state = self.random_start.sample()[:self.n_obs] if self.random_start is not False else self.initial_state
        self._set_robot_state(self.initial_state)

        # Change target in case of multiple targets.
        _target = self.goals[torch.randint(low=0, high=len(self.goals), size=(1,)).item()]
        self._set_target(_target.numpy())

        # Get goal representation and distribution q.
        if self.goal_conditioned_policy or self.reward_type == 'edl':
            x_hat, x_hat_var, latent, _ = self.vae(torch.tensor(self.target, device=self.device).float(), self.device, True, True)
            self.condition = latent if self.latent_rep else self.target
            if self.reward_type == 'edl':
                # TODO: Check if reparam should be used here.
                # x_hat, x_hat_var, latent, _ = self.vae(torch.tensor(self.target, device=self.device).float(), self.device, False, True)
                self.dist = torch.distributions.MultivariateNormal(x_hat.cpu(), torch.diag(x_hat_var.exp().cpu()))
        return self._observations()

    def step(self, action):
        self.iterations += 1
        self._robot_act(action)
        observation = self._observations()
        reward = self._reward_fn(observation)
        done = False
        if self.max_steps == self.iterations:
            done = True
        if self.graphics:
            self.render()
        return observation, reward, done, {}

    def _reward_fn(self, observation):
        if self.reward_type == 'mse':
            reward = torch.linalg.norm(observation[:self.n_obs] - self.target, ord=2)
            return -reward
        elif self.reward_type == 'edl':
            scaled_obs = torch.tensor(self.scaler(observation[:self.n_obs])) if self.scaler is not None else observation
            reward = self.dist.log_prob(scaled_obs[:self.n_obs]).cpu().item()
            return reward

    def set_max_steps(self, max_steps):
        self.max_steps = max_steps

    def _set_target(self, goal):
        self.target = goal

    def render(self): ...
    def close(self): ...
    def _observations(self): ...
    def _set_robot_state(self, state): ...
    def _robot_act(self, action): ...
    def _state(self): ...
    def _reward_fn(self, observation): ...
    def _reset_op(self): ...
