import os
import pickle
import json
import numpy as np
import argparse
import torch
from tqdm import trange, tqdm
from ._td3 import TD3


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

def train_td3(env, actor_net, critic_net):
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--episodes", default=1e4, type=int)
    parser.add_argument("--start-episode", default=20, type=int)   # Time steps initial random policy is used
    parser.add_argument("--eval-freq", default=1e+5, type=int)       # How often (time steps) we evaluate
    parser.add_argument("--actor-lr", default=1e-3, type=float)                # Actor learning rate
    parser.add_argument("--critic-lr", default=1e-3, type=float)                # Critic learning rate
    parser.add_argument("--expl-noise", default=0.1, type=float)                # Std of Gaussian exploration noise
    parser.add_argument("--batch-size", default=256, type=int)      # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)                 # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)                     # Target network update rate
    parser.add_argument("--policy-noise", default=0.2, type=float)              # Noise added to target policy during critic update
    parser.add_argument("--noise-clip", default=0.5, type=float)                # Range to clip target policy noise
    parser.add_argument("--policy-freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--file-name", type=str)
    parser.add_argument("-g", "--graphics", default=False, action='store_true')
    parser.add_argument("--checkpoint-episodes", default=1e+3, type=int)
    args = parser.parse_args()
    args.start_episode *= args.steps
    args.eval_freq *= args.steps
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    file_name = args.file_name
    tq = tqdm()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {env.name}, Seed: {args.seed}")
    print("---------------------------------------")

    metadata = {
        'algorithm': args.policy,
        'steps': args.steps,
    }
    if not os.path.exists(file_name):
        os.mkdir(file_name)
    print("Saving env.")
    with open(f"{file_name}/env.pickle", 'wb') as env_file:
        pickle.dump(env, env_file)
    print("Saving metadata.")
    with open(f"{file_name}/metadata.json", 'w') as metadata_file:
        json.dump(metadata, metadata_file)

    if not os.path.exists(f'{file_name}/logs'):
        os.mkdir(f'{file_name}/logs')

    if args.graphics:
        env.render()

    # Set seeds
    if args.seed > 0:
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    env.set_max_steps(args.steps)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    kwargs = {
        'actor_net': actor_net,
        'critic_net': critic_net,
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


    replay_buffer = ReplayBuffer(state_dim, action_dim)

    # Evaluate untrained policy
    evaluations = []

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    prob_same_init = 0.99
    prob_same = prob_same_init

    for t in trange(int(args.steps * args.episodes)):

        episode_timesteps += 1

        if ((t+1) % (env.max_steps * 5000)) == 0:
            # Explore more.
            args.start_episode = t + (1000*args.steps)
            decay = 0.90
            action_weight_decay = 1.005
            env.action_weight = min(1.5, action_weight_decay * env.action_weight)
            env.sigma_sq = max(0.1,decay*env.sigma_sq)
            for idx in range(len(replay_buffer.reward)):
                replay_buffer.reward[idx] = env._reward_fn(replay_buffer.state[idx] * 600, replay_buffer.action[idx])
            with open(f"{file_name}/env.pickle", 'wb') as env_file:
                pickle.dump(env, env_file)

        p = np.random.uniform(0., 1.)
        # Select action randomly or according to policy
        if t < args.start_episode:
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
        next_state, reward, done, _ = env.step(action)
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
            tq.set_description(desc=f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            if ((t+1) % (args.checkpoint_episodes * args.steps)) == 0:
                # checkpoint = {
                #     'actor_model': policy.actor_target,
                #     'actor_state_dict': policy.actor_target.state_dict(),
                #     'actor_optimizer' : policy.actor_optimizer.state_dict(),
                #     'critic_model': policy.critic_target,
                #     'critic_state_dict': policy.critic_target.state_dict(),
                #     'critic_optimizer' : policy.critic_optimizer.state_dict(),
                # }
                # torch.save(checkpoint, f'{file_name}/logs/policy_{t+1}_steps.pth')
                with open(f'{file_name}/logs/policy_replay_buffer_{t+1}_steps.pickle', 'wb') as rb_file:
                    pickle.dump(replay_buffer, rb_file)

                with open(f'{file_name}/logs/policy_{t+1}_steps.pickle', 'wb') as policy_file:
                    pickle.dump(policy, policy_file)
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

            if t >= args.start_episode:
                for _ in range(20):
                    policy.train(replay_buffer, args.batch_size)

        # # Evaluate episode
        # if (t + 1) % args.eval_freq == 0:
        #     evaluations.append(eval_policy(policy, env, args.seed, graphics=True))
    with open(f'{file_name}/policy.pickle', 'wb') as policy_file:
        pickle.dump(policy, policy_file)

def eval_policy(policy, env, seed, eval_episodes=2, graphics=False):
    eval_env = env
    env.reset()
    if graphics:
        env.render()
    avg_reward = 0.
    s = None
    if seed > 0:
        s = seed + 100
    for _ in range(eval_episodes):
        state, done = eval_env.reset(seed=s), False
        while not done:
            action = policy.select_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    env.close()
    env.reset()
    return avg_reward