import numpy as np
import torch
import gym
import pyfastsim as fastsim

class KheperaEnv(gym.Env):

    def __init__(
        self,
        enable_graphics = False,
        *,
        radius = 20,
        posture = [50, 50, 0],
        map_path = "../data_collection/worlds/three_wall.pbm",
        reward_dist = None,
        min_max_scale = None,
        max_steps = 0
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
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, -np.pi]), high=np.array([600, 600, np.pi]), shape=(3,), dtype=np.float32)
        self.ep_reward = 0

    def step(self, action, *, eval=False):
        self.it += 1
        if self.enable_graphics:
            self.disp.update()
        self.robot.move(*action, self.map, False)
        observation = self.__observation()
        scaled_obs = torch.tensor((observation - self.min)/(self.max - self.min))
        # reward = self.dist.log_prob(scaled_obs).cpu().item()
        reward = np.linalg.norm(observation[:2] - self.target[:2], ord=2)
        done = False
        if eval:
            if np.linalg.norm(observation[:2] - self.target[:2], ord=2) == 0:
                done = True
        else:
            if np.linalg.norm(observation[:2] - self.target[:2], ord=2) == 0 or self.max_steps == self.it:
                done = True

        return np.array(observation, dtype=np.float32), -reward, done, {}

    def set_target(self, target_pos):
        self.target = target_pos
        if self.enable_graphics:
            self.map.add_goal(fastsim.Goal(*self.target[:2], 20, 1))
            self.disp.update()

    def reset(self):
        self.it = 0
        self.robot.set_pos(self.posture)
        self.robot.move(0, 0, self.map, False)
        if self.enable_graphics:
            self.disp.update()
        return np.array(self.__observation(), dtype=np.float32)

    def render(self):
        self.disp = fastsim.Display(self.map, self.robot)
        self.disp.update()
        self.enable_graphics = True

    def __observation(self):
        _rpos = self.robot.get_pos()
        return [_rpos.x(), _rpos.y(), _rpos.theta()]