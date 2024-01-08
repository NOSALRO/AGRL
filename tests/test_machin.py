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
import numpy as np
import torch
from agrl.env import KheperaEnv, Box
from machin.frame.algorithms import TD3
from agrl.vae import StatesDataset, VariationalAutoencoder, train, visualize
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import pyfastsim as fastsim
from machin.frame.algorithms import PPO
from machin.utils.logging import default_logger as logger



dataset = StatesDataset(path="data/no_wall.dat")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Env Init.
world_map = fastsim.Map('worlds/no_wall.pbm', 600)
robot = fastsim.Robot(10, fastsim.Posture(100., 100., 0.))

action_space = Box(low=-1., high=1., shape=(2,), dtype=np.float32)
observation_space = Box(
    low=np.array([0, 0, -np.pi]),
    high=np.array([600, 600, np.pi]),
    shape=(3,),
    dtype=np.float32
)
start_space = Box(
    low=np.array([25, 25, -np.pi]),
    high=np.array([575, 575, np.pi]),
    shape=(3,),
    dtype=np.float32
)

env = KheperaEnv(
    world_map=world_map,
    robot=robot,
    reward_type='mse',
    n_obs=4,
    goals=[500, 200, 0],
    goal_conditioned_policy=False,
    latent_rep=False,
    observation_space=observation_space,
    action_space=action_space,
    random_start=False,
    max_steps=1000,
    vae=None,
    scaler=None
)

# configurations
observe_dim = 4
action_dim = 2
max_episodes = 10000
max_steps = 1000
noise_param = (0, 0.2)
noise_mode = "normal"
solved_reward = -100
solved_repeat = 10

# model definition
# model definition
class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = torch.nn.Linear(state_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, action_dim)

    def forward(self, state):
        a = torch.tanh(self.fc1(state))
        a = torch.tanh(self.fc2(a))
        a = torch.tanh(self.fc3(a))
        return a


class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = torch.nn.Linear(state_dim + action_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q = torch.relu(self.fc1(state_action))
        q = torch.relu(self.fc2(q))
        q = self.fc3(q)
        return q




if __name__ == "__main__":

    actor = Actor(observe_dim, action_dim)
    actor_t = Actor(observe_dim, action_dim)
    critic = Critic(observe_dim, action_dim)
    critic_t = Critic(observe_dim, action_dim)
    critic2 = Critic(observe_dim, action_dim)
    critic2_t = Critic(observe_dim, action_dim)

    td3 = TD3(
        actor,
        actor_t,
        critic,
        critic_t,
        critic2,
        critic2_t,
        torch.optim.Adam,
        torch.nn.MSELoss(reduction="sum"),
    )


    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while episode < max_episodes:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = torch.tensor(env.reset()[0], dtype=torch.float32).view(1, observe_dim)
        if episode % 100 == 0:
            env.render()
        else:
            env.close()

        tmp_observations = []
        while not terminal and step <= max_steps:
            step += 1
            with torch.no_grad():
                old_state = state
                # agent model inference
                # action = td3.act_with_noise(
                #     {"state": old_state}, noise_param=noise_param, mode=noise_mode
                # )
                action = td3.act({"state": old_state})
                state, reward, terminal, _, _ = env.step(action.detach().cpu().numpy()[0])
                state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
                total_reward += reward

                tmp_observations.append(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )

        # update
        td3.store_episode(tmp_observations)
        if episode > 100:
            for _ in range(step):
                td3.update()


        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        if smoothed_total_reward > solved_reward:
            reward_fulfilled += 1
            if reward_fulfilled >= solved_repeat:
                logger.info("Environment solved!")
                exit(0)
        else:
            reward_fulfilled = 0