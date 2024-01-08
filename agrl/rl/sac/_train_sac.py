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
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.ppo import MlpPolicy
from ..utils import cli


def train_sac(env, device):
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
    policy_kwargs = dict(net_arch=dict(pi = [32, 32], qf = [32,32]))
    sac_args = dict(
        learning_rate=args.actor_lr,
        buffer_size=100000,
        learning_starts=args.start_episode,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.discount,
        train_freq=(args.policy_freq, 'episode'),
        gradient_steps=args.epochs,
        replay_buffer_class=None,
        replay_buffer_kwargs=None,
        optimize_memory_usage=False,
        ent_coef='auto',
        target_update_interval=1,
        target_entropy='auto',
        use_sde=False,
        sde_sample_freq=-1,
        tensorboard_log=None,
        use_sde_at_warmup=True,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=None,
    )
    model = SAC('MlpPolicy', env, action_noise=NormalActionNoise(0, args.expl_noise), device=device, **sac_args)
    metadata = dict(algorithm='sac')
    metadata['steps'] = args.steps
    metadata['episodes'] = args.episodes
    metadata['expl_noise'] = args.expl_noise
    metadata['algorithm_args'] = sac_args
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
        save_freq=args.checkpoint_episodes * args.steps,
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
