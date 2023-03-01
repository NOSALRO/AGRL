import numpy as np
import torch
import gym
import pyfastsim as fastsim

class KheperaEnv(gym.Env):

    def __init__(
        self,
        *,
        radius = 20,
        posture = [50, 50, 0],
        map_path = "../data_collection/worlds/three_wall.pbm",
        reward_dist = None,
        min_max_scale = None,
        max_steps = 0,
        goals = None,
        vae = None,
    ):
        super().__init__()
        self.dist = reward_dist
        self.enable_graphics = False
        self.radius = radius
        self.posture = fastsim.Posture(*posture)
        self.map = fastsim.Map(map_path, 600)
        self.robot = fastsim.Robot(radius, self.posture)
        self.max_steps = max_steps
        self.min, self.max = min_max_scale
        self.action_space = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, -1, -1, -np.inf, -np.inf]), high=np.array([600, 600, 1, 1, np.inf, np.inf]), shape=(6,), dtype=np.float32)
        self.goals = goals
        self.goal_idx = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else "cpu")
        self.vae = vae.to(self.device)
        self.map_path = map_path

    def step(self, action, *, eval=False):
        self.it += 1
        if self.enable_graphics:
            self.disp.update()
        self.robot.move(*action, self.map, False)
        observation = self.__observation()
        # scaled_obs = torch.tensor((observation - self.min)/(self.max - self.min))
        # reward = self.dist.log_prob(scaled_obs[:4]).cpu().item()
        reward = np.linalg.norm(observation[:2] - self.target[:2], ord=2)
        done = False
        if eval:
            if np.linalg.norm(observation[:2] - self.target[:2], ord=2) == 0:
                done = True
        else:
            if self.max_steps == self.it:
                done = True
        return observation, -reward, done, {}

    def set_target(self, target_pos):
        self.target = target_pos
        if self.enable_graphics:
            self.map.add_goal(fastsim.Goal(*self.target[:2], 20, 1))
            self.disp.update()

    def reset(self):
        # target = self.goals[np.random.randint(0, len(self.goals), size=1)][0]
        goal = [1000, 8000]
        self.goal_idx = not self.goal_idx
        target = self.goals[goal[self.goal_idx]]
        x_hat, x_hat_var, self.latent, _ = self.vae(torch.tensor(target, device=self.device).float(), self.device, True, True)
        self.set_target(target)
        self.dist = torch.distributions.MultivariateNormal(x_hat.cpu(), torch.diag(x_hat_var.mul(.5).exp().cpu()))

        self.it = 0
        self.posture = fastsim.Posture(*np.random.uniform(low=50, high=550, size=2), 0)
        self.robot.set_pos(self.posture)
        self.robot.move(0, 0, self.map, False)
        if self.enable_graphics:
            self.disp.update()
        return self.__observation()

    def render(self):
        self.disp = fastsim.Display(self.map, self.robot)
        self.disp.update()
        self.enable_graphics = True

    def close(self):
        del self.disp
        self.enable_graphics = False

    def __observation(self):
        _rpos = self.robot.get_pos()
        return np.array([_rpos.x(), _rpos.y(), np.cos(_rpos.theta()), np.sin(_rpos.theta()), *self.latent.cpu().detach().numpy()], dtype=np.float32)
