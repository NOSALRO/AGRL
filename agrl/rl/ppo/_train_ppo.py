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
