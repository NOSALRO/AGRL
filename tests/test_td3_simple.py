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
import copy
import argparse
import os
import numpy as np
import torch
import numpy as np
import torch.nn.functional as F
# from agrl.env import KheperaEnv, Box
from agrl.env import Box
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
from agrl.controllers import Controller, DVController
import pyfastsim as fastsim
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class KheperaEnv:
    def __init__(self, max_steps, graphics = False):
        self.map = fastsim.Map('worlds/no_wall.pbm', 600)
        self.init_posture = fastsim.Posture(500., 500., 0.)
        self.target = fastsim.Posture(170., 160., 0.)
        self.robot = fastsim.Robot(10, self.init_posture)
        self.action_space = Box(low=np.array([-np.pi, -1.]), high=np.array([np.pi, 1.]), shape=(2,), dtype=np.float32)
        self.observation_space = Box(
            low=np.array([0, 0, -1, -1]),
            high=np.array([1., 1., 1, 1]),
            shape=(4,),
            dtype=np.float32
        )
        self.controller = DVController()
        self.sigma_sq = 100.
        self.action_weight = 0.001

        self.iterations = 0
        self.max_steps = max_steps
        self.graphics = graphics
        self.n_obs = 4

    def _obs(self):
        p = self.robot.get_pos()
        return np.array([[p.x()/600., p.y()/600., np.cos(p.theta()), np.sin(p.theta())]])

    def reset(self, seed = 0):
        self.robot.set_pos(self.init_posture)
        self.iterations = 0

        return self._obs()

    def reward(self, state, action):
        act = np.linalg.norm(action)
        dist = np.linalg.norm(state[:2] - [self.target.get_x(), self.target.get_y()])
        return np.exp(-dist/self.sigma_sq) - self.action_weight * act

    def step(self, action):
        self.iterations += 1
        # action = self.action_space.clip(action)

        # p_theta = self.robot.get_pos().theta()


        # self.action_space.unscale(np.array([-1,1]), -1, 1)
        cmd = self.controller.update(action)
        for _ in range(10):
            self.robot.move(cmd[0], cmd[1], self.map, False)
            if self.graphics:
                self.render()

        # c_theta = self.robot.get_pos().theta()

        dist = self.robot.get_pos().dist_to(self.target)# / float(self.max_steps)
        act = np.linalg.norm(action)# / float(self.max_steps)

        # theta_diff = c_theta - p_theta
        # while theta_diff < -2.*np.pi:
        #     theta_diff += 2.*np.pi
        # while theta_diff > 2.*np.pi:
        #     theta_diff -= 2.*np.pi

        # total_reward = -dist - 0.1 * act
        # total_reward = -act*act
        # total_reward = - dist
        # total_reward = np.exp(-dist/sigma_sq) - 0.01 * act * act - theta_diff * theta_diff
        total_reward = np.exp(-dist/self.sigma_sq) - self.action_weight * act
        # print(total_reward, self.reward(np.array([self.robot.get_pos().x(), self.robot.get_pos().y()]), cmd))

        done = False
        if self.graphics:
            self.render()

        if self.iterations == self.max_steps:
            done = True
            # print(self.robot.get_pos().x(), self.robot.get_pos().y(), self.robot.get_pos().theta(), "vs", self.target.x(), self.target.y(), self.target.theta())
            # if dist < 30. / float(self.max_steps):
            #     total_reward += 200.

        return self._obs(), total_reward, done, cmd

    def render(self):
        if not hasattr(self, 'disp'):
            self.disp = fastsim.Display(self.map, self.robot)
            self.graphics = True
        self.map.clear_goals()
        self.map.add_goal(fastsim.Goal(self.target.x(), self.target.y(), 10., 1))
        self.disp.update()

    def close(self):
        if hasattr(self, 'disp'):
            del self.disp
        self.map.clear_goals()
        self.graphics = False

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = torch.nn.Linear(state_dim, 64)
        self.l2 = torch.nn.Linear(64, 64)
        self.l3 = torch.nn.Linear(64, action_dim)
        self.max_action = max_action


    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Q1 architecture
        self.l1 = torch.nn.Linear(state_dim + action_dim, 256)
        self.l2 = torch.nn.Linear(256, 256)
        self.l3 = torch.nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = torch.nn.Linear(state_dim + action_dim, 256)
        self.l5 = torch.nn.Linear(256, 256)
        self.l6 = torch.nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = torch.relu(self.l4(sa))
        q2 = torch.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = torch.relu(self.l1(sa))
        q1 = torch.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        actor_lr=1e-3,
        critic_lr=1e-3
    ):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env, seed, eval_episodes=1):
    eval_env = env
    env.reset()
    env.render()
    avg_reward = 0.
    s = None
    if seed > 0:
        s = seed + 100
    for _ in range(eval_episodes):
        state, done = eval_env.reset(seed=s), False
        t = 0
        # prob_same = prob_same_init = 0.9
        while not done:
            action = policy.select_action(np.array(state))
            # if p >= prob_same or t == 0:
            #     action = env.action_space.sample()
            #     prob_same = prob_same_init
            # else:
            #     prob_same = prob_same * prob_same
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward
            t += 1

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    env.close()
    env.reset()
    return avg_reward

if not os.path.exists("./.tmp/td3/results"):
    os.makedirs("./.tmp/td3/results")

if True and not os.path.exists("./.tmp/td3/models"):
    os.makedirs("./.tmp/td3/models")

# # Env Init.
# world_map = fastsim.Map('worlds/no_wall.pbm', 600)
# robot = fastsim.Robot(10, fastsim.Posture(500., 500., 0.))

# action_space = Box(low=-1., high=1., shape=(2,), dtype=np.float32)
# observation_space = Box(
#     low=np.array([0, 0, -1, -1]),
#     high=np.array([600, 600, 1, 1]),
#     shape=(4,),
#     dtype=np.float32
# )
# start_space = Box(
#     low=np.array([25, 25, -np.pi]),
#     high=np.array([575, 575, np.pi]),
#     shape=(3,),
#     dtype=np.float32
# )

# env = KheperaEnv(
#     robot=robot,
#     world_map=world_map,
#     reward_type='mse',
#     n_obs=4,
#     goals=[100, 170, 0],
#     goal_conditioned_policy=False,
#     latent_rep=False,
#     observation_space=observation_space,
#     action_space=action_space,
#     random_start=False,
#     max_steps=600,
#     vae=None,
#     scaler=None,
# )
env = KheperaEnv(max_steps=100)
parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
parser.add_argument("--env", default="KheperaController")          # OpenAI gym environment name
parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=100*20, type=int)# Time steps initial random policy is used
parser.add_argument("--eval_freq", default=100*50, type=int)       # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
parser.add_argument("--actor_lr", default=1e-3, type=float)                # Actor learning rate
parser.add_argument("--critic_lr", default=1e-3, type=float)                # Critic learning rate
parser.add_argument("--expl_noise", default=0.1, type=float)                # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
parser.add_argument("--policy_noise", default=0.2, type=float)              # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5, type=float)                # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
args = parser.parse_args()

file_name = f"{args.policy}_{args.env}_{args.seed}"
print("---------------------------------------")
print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
print("---------------------------------------")


# Set seeds
if args.seed > 0:
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
# max_action = env.action_space.high[0]
max_action = 1

kwargs = {
    "state_dim": env.n_obs,
    "action_dim": 2,
    "max_action": max_action,
}

# Initialize policy
if args.policy == "TD3":
    # Target policy smoothing is scaled wrt the action scale
    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    kwargs["actor_lr"] = args.actor_lr
    kwargs["critic_lr"] = args.critic_lr
    policy = TD3(**kwargs)
elif args.policy == "OurDDPG":
    policy = OurDDPG.DDPG(**kwargs)
elif args.policy == "DDPG":
    policy = DDPG.DDPG(**kwargs)

if args.load_model != "":
    policy_file = file_name if args.load_model == "default" else args.load_model
    policy.load(f"./.tmp/td3/models/{policy_file}")

replay_buffer = ReplayBuffer(state_dim, action_dim)

# Evaluate untrained policy
# evaluations = [eval_policy(policy, env, args.seed)]
evaluations = []

state, done = env.reset(), False
episode_reward = 0
episode_timesteps = 0
episode_num = 0

prob_same_init = 0.99
prob_same = prob_same_init

for t in range(int(args.max_timesteps)):

    episode_timesteps += 1

    if ((t+1) % 30000) == 0:
        decay = 0.9
        action_weight_decay = 5
        env.action_weight = action_weight_decay * env.action_weight
        env.sigma_sq = min(0.5,decay*env.sigma_sq)
        for idx in range(len(replay_buffer.reward)):
            replay_buffer.reward[idx] = env.reward(replay_buffer.state[idx] * 600, replay_buffer.action[idx])

    p = np.random.uniform(0., 1.)
    # Select action randomly or according to policy
    if t < args.start_timesteps:
        if p >= prob_same or t == 0:
           action = env.action_space.sample()
           prob_same = prob_same_init
        else:
            prob_same = prob_same * prob_same
    else:
        action = (
            policy.select_action(np.array(state))
            + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
        ).clip(-max_action, max_action)

    # Perform action
    next_state, reward, done, act = env.step(action)
    done_bool = float(done) if episode_timesteps < env.max_steps else 0

    # Store data in replay buffer
    replay_buffer.add(state, action, next_state, reward, done_bool)

    state = next_state
    episode_reward += reward

    # # Train agent after collecting sufficient data
    # if t >= args.start_timesteps:
    #     policy.train(replay_buffer, args.batch_size)

    if done:
        # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
        print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
        # Reset environment
        state, done = env.reset(), False
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1

        if t >= args.start_timesteps:
            for _ in range(20):
                policy.train(replay_buffer, args.batch_size)

    # Evaluate episode
    if (t + 1) % args.eval_freq == 0:
        print(np.linalg.norm(torch.nn.utils.parameters_to_vector(policy.actor.parameters()).cpu().detach().numpy()))
        evaluations.append(eval_policy(policy, env, args.seed))
        np.save(f"./.tmp/td3/results/{file_name}", evaluations)
        if args.save_model: policy.save(f"./.tmp/td3/models/{file_name}")