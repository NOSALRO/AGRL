import os
import sys
import time
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from nosalro.rl import cli


def learn(env, device):
    STEPS = 1e+04
    EPISODES = 1e+04
    algorithm, mode, graphics, file_name, STEPS, EPISODES = cli(STEPS, EPISODES)
    env.set_max_steps(STEPS)
    env.reset()
    if graphics:
        env.render()
    # Set up RL algorithm.
    if algorithm.lower() == 'sac':
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=1e-4,
            buffer_size=1000000,
            learning_starts=100,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, 'episode'),
            gradient_steps=100,
            action_noise=NormalActionNoise(0, 0.05),
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
            policy_kwargs=None,
            verbose=1,
            seed=None,
            device=device,
        )
    elif algorithm.lower() == 'ppo':
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=512,
            n_epochs=80,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=False,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=1,
            target_kl=None,
            policy_kwargs=None,
            verbose=1,
            seed=None,
            device=device
        )

    # Set up mode.
    if mode.lower() == 'train':
        try:
            if not os.path.exists(file_name):
                os.mkdir(file_name)
            checkpoint_callback = CheckpointCallback(
            save_freq=1e+05,
            save_path=f"{file_name}/logs/",
            name_prefix=file_name.split('/')[-1],
            save_replay_buffer=True,
            save_vecnormalize=True,
            verbose=True
            )
            model.learn(total_timesteps=STEPS*EPISODES, callback=checkpoint_callback, progress_bar=True)
        except KeyboardInterrupt:
            print("Training stopped!")

        model.save(f"{file_name}/{file_name.split('/')[-1]}")
        print(f"Saving model at {file_name}/{file_name.split('/')[-1]}.zip")

    elif mode.lower() == 'eval' or mode.lower() == 'continue':
        print(f"Loading model {file_name}.zip")
        if algorithm.lower() == 'sac':
            model = SAC.load(f"{file_name}/{file_name.split('/')[-1]}.zip", env=env)
        elif algorithm.lower() == 'ppo':
            model = PPO.load(f"{file_name}/{file_name.split('/')[-1]}.zip", env=env)

    if mode.lower() == 'continue':
        try:
            model = model.learn(total_timesteps=STEPS *
                                EPISODES, progress_bar=True)
        except KeyboardInterrupt:
            print("Training stopped!")
            model.save(f"{file_name}/{file_name.split('/')[-1]}")
            print(f"Saving model at {file_name}/{file_name.split('/')[-1]}.zip")

    if not graphics:
        observation = env.reset()
        env.render()
        while True:
            try:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, done, _ = env.step(action)
                if done:
                    time.sleep(3)
                    print('Env Reseted')
                    env.reset()
            except KeyboardInterrupt:
                print("Exit")
                return model
