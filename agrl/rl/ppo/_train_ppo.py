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
import os
import pickle
import json
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.ppo import MlpPolicy
from ..utils import cli


def train_ppo(env, device):
    args = cli()
    if not os.path.exists(args.file_name):
        os.mkdir(args.file_name)
    with open(f"{args.file_name}/env.pickle", 'wb') as env_file:
        print("Saving env.")
        pickle.dump(env, env_file)

    env.set_max_steps(args.steps)
    env.reset()
    if args.graphics:
        env.render()
    # Set up RL algorithm.
    policy_kwargs = dict(squash_output = True)
    ppo_args = dict(
        learning_rate=args.actor_lr,
        n_steps=args.update_steps,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        gamma=args.discount,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=1,
        target_kl=None,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=None,
    )
    model = PPO('MlpPolicy', env, device=device, **ppo_args)
    metadata = dict(algorithm='ppo')
    metadata['steps'] = args.steps
    metadata['episodes'] = args.episodes
    metadata['algorithm_args'] = ppo_args
    metadata['call_fname'] = 'predict'
    metadata['fargs'] = {'deterministic': True}
    with open(f"{args.file_name}/metadata.json", 'w') as metadata_file:
        print("Saving metadata.")
        json.dump(metadata, metadata_file)

    # Set up mode.
    try:
        if not os.path.exists(args.file_name):
            os.mkdir(args.file_name)
        checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_episodes,
        save_path=f"{args.file_name}/logs/",
        name_prefix='policy',
        save_replay_buffer=True,
        save_vecnormalize=True,
        verbose=True
        )
        model.learn(total_timesteps=args.steps*args.episodes, callback=checkpoint_callback, progress_bar=True)
    except KeyboardInterrupt:
        print("Training stopped!")

    model.save(f"{args.file_name}/policy")
    print(f"Saving model at {args.file_name}")
