import time
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise
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
            train_freq=1,
            gradient_steps=1,
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
            learning_rate=1e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=10,
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
            model = model.learn(total_timesteps=STEPS *
                                EPISODES, progress_bar=True)
        except KeyboardInterrupt:
            print("Training stopped!")

        model.save(file_name)
        print(f"Saving model at {file_name}.zip")

    elif mode.lower() == 'eval' or mode.lower() == 'continue':
        print(f"Loading model {file_name}.zip")
        if algorithm.lower() == 'sac':
            model = SAC.load(file_name + '.zip', env=env)
        elif algorithm.lower() == 'ppo':
            model = PPO.load(file_name + '.zip', env=env)

    if mode.lower() == 'continue':
        try:
            model = model.learn(total_timesteps=STEPS *
                                EPISODES, progress_bar=True)
        except KeyboardInterrupt:
            print("Training stopped!")
            model.save(file_name)
            print(f"Saving model at {file_name}.zip")

    if not graphics:
        observation = env.reset()
        env.render()
        while True:
            try:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, done, _ = env.step(action)
                if done:
                    time.sleep(3)
                    env.reset()
                    env.render()
            except KeyboardInterrupt:
                print("Exit")
                return model
