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
