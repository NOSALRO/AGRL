import torch
import gym
import numpy as np
import copy


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
        random_start = False,
        goal_conditioned_policy = False,
        latent_rep = True,
        min_max = None,
        goals = None,
        vae = None,
        target = None,
    ):
        # TODO: specify columns' index coordinates to calculate MSE, e.g. (x, y, z). Workaround: overload reward function.
        super().__init__()
        self.robot = copy.deepcopy(robot)
        self.target = target if target else False
        self.reward_type = reward_type.lower()
        self.n_obs = n_obs
        self.max_steps = max_steps
        self.goal_conditioned_policy = goal_conditioned_policy
        self.goals = goals
        self.latent_rep = latent_rep
        self.graphics = False

        assert isinstance(observation_space, gym.spaces.Box), "observation_space type must be gym.spaces.Box"
        assert isinstance(action_space, gym.spaces.Box), "action_space type must be gym.spaces.Box"
        self.action_space = action_space
        self.observation_space = observation_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.min, self.max = None, None
        if min_max is not None:
            self.min, self.max = min_max
        if vae is not None:
            self.vae = vae.to(self.device)
        if self.goals is None:
            self._set_target(self.target)

        # Incase starting position should be bounded.
        if isinstance(random_start, gym.spaces.Box):
            self.random_start = random_start
        elif random_start:
            self.random_start = self.observation_space
        else:
            self.random_start = self.initial_state

    def reset(self):
        self.iterations = 0
        self.initial_state = self.random_start.sample()[:self.n_obs] if self.random_start is not False else self.initial_state
        self._set_robot_state(self.initial_state)
        if self.goals is not None:
            _target = self.goals[np.random.randint(low=0, high=len(self.goals), size=1).item()]
            self._set_target(_target)
        if self.goal_conditioned_policy or self.reward_type == 'edl':
            x_hat, x_hat_var, latent, _ = self.vae(torch.tensor(self.target, device=self.device).float(), self.device, True, True)
            self.condition = latent if self.latent_rep else self.target
            if self.reward_type == 'edl':
                self.dist = torch.distributions.MultivariateNormal(x_hat.cpu(), torch.diag(x_hat_var.mul(.5).exp().cpu()))
        return self._observations()

    def step(self, action, eval=False):
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
            reward = np.linalg.norm(observation[:self.n_obs] - self.target, ord=2)
            return -reward
        elif self.reward_type == 'edl':
            scaled_obs = torch.tensor((observation - self.min)/(self.max - self.min)) if self.min is not None else observation
            reward = self.dist.log_prob(scaled_obs[:self.n_obs]).cpu().item()
            return reward

    def set_max_steps(self, max_steps):
        self.max_steps = max_steps