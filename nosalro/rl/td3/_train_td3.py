import copy
import os
import pickle
import json
import numpy as np
import argparse
import torch
from tqdm import trange, tqdm
from ..utils import cli
from ._td3 import TD3
from ._replay_buffer import ReplayBuffer
from ..utils import eval_policy


def train_td3(env, actor_net, critic_net, eval_data=None):
    tq = tqdm()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = cli()

    metadata = dict(algorithm='td3')
    metadata['steps'] = args.steps
    metadata["policy_noise"] = args.policy_noise
    metadata["noise_clip"] = args.noise_clip
    metadata["policy_freq"] = args.policy_freq
    metadata["actor_lr"] = args.actor_lr
    metadata["critic_lr"] = args.critic_lr
    metadata["tau"] = args.tau
    metadata["epochs"] = args.epochs
    metadata["seed"] = args.seed
    metadata["expl_noise"] = args.expl_noise
    metadata["start_episode"] = args.start_episode
    metadata["eval_freq"] = args.eval_freq

    if not os.path.exists(args.file_name):
        os.mkdir(args.file_name)

    print("Saving env.")
    with open(f"{args.file_name}/env.pickle", 'wb') as env_file:
        pickle.dump(env, env_file)

    print("Saving metadata.")
    with open(f"{args.file_name}/metadata.json", 'w') as metadata_file:
        json.dump(metadata, metadata_file)

    if not os.path.exists(f'{args.file_name}/logs'):
        os.mkdir(f'{args.file_name}/logs')


    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)

    env.set_max_steps(args.steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # Initialize policy
    policy = TD3(actor_net=actor_net, critic_net=critic_net, **metadata)
    replay_buffer = ReplayBuffer(state_dim, action_dim, device)

    # Evaluate untrained policy
    evaluations = []

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    if args.graphics:
        env.reset()
        env.render()
    for t in trange(args.steps * args.episodes):

        episode_timesteps += 1

        # if (t % (env.max_steps * 5000)) == 0 and t>0:
        #     # Explore more.
        #     args.start_episode = t + (1000*args.steps)
        #     decay = 0.95
        #     env.sigma_sq = max(0.01,decay*env.sigma_sq)
        #     action_weight_decay = 1.1
        #     env.action_weight = min(2, action_weight_decay * env.action_weight)
        #     for idx in range(len(replay_buffer.reward)):
        #         replay_buffer.reward[idx] = env._reward_fn(replay_buffer.state[idx], replay_buffer.action[idx])
        #     with open(f"{args.file_name}/env.pickle", 'wb') as env_file:
        #         pickle.dump(env, env_file)

        p = np.random.uniform(0., 1.)
        # Select action randomly or according to policy
        if t < (args.start_episode * args.steps):
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        # Perform action
        next_state, reward, done, _ = env.step(action)
        done_bool = float(done) if episode_timesteps < env.max_steps else 0

        # Store data in replay buffer
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        if done:
            tq.set_description(desc=f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            if ((t+1) % (args.checkpoint_episodes * args.steps)) == 0:
                checkpoint = {
                    'actor_model': policy.actor_target,
                    'actor_state_dict': policy.actor_target.state_dict(),
                    'actor_optimizer' : policy.actor_optimizer.state_dict(),
                    'critic_model': policy.critic_target,
                    'critic_state_dict': policy.critic_target.state_dict(),
                    'critic_optimizer' : policy.critic_optimizer.state_dict(),
                }
                # torch.save(checkpoint, f'{args.file_name}/logs/policy_{t+1}_steps.pth')

                with open(f'{args.file_name}/logs/policy_replay_buffer_{t+1}_steps.pickle', 'wb') as rb_file:
                    pickle.dump(replay_buffer, rb_file)
                with open(f'{args.file_name}/logs/policy_{t+1}_steps.pickle', 'wb') as policy_file:
                    pickle.dump(policy, policy_file)

            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            if t >= (args.start_episode * args.steps):
                for _ in range(args.epochs):
                    policy.train(replay_buffer, args.batch_size)

            # # Evaluate episode
            if ((t + 1) % (args.eval_freq * args.steps)) == 0 and eval_data is not None:
                if args.graphics:
                    env.close()
                evaluations.append(eval_policy(policy, env, args.seed, eval_data, args.graphics))
                if args.graphics:
                    env.render()

    with open(f'{args.file_name}/policy.pickle', 'wb') as policy_file:
        pickle.dump(policy, policy_file)
    with open(f'{args.file_name}/policy_replay_buffer.pickle', 'wb') as rb_file:
        pickle.dump(replay_buffer, rb_file)