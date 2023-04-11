import numpy as np
from scipy.spatial.transform import Rotation
from nosalro.robots import Anymal
from nosalro.env.anymal import AnymalEnv
from nosalro.env import Box
from nosalro.rl.td3 import Actor, Critic, train_td3, eval_policy


anymal = Anymal()
action_lim = 2*anymal._mass
action_space = Box(
    low=np.zeros(4*3*1,),
    high=np.multiply(np.ones(4*3*1,), action_lim),
    shape=(4*3*1, ),
    dtype=np.float32
)

observation_space = Box(
    low = np.array([-np.inf, -np.inf, 0, -np.pi, -np.pi, -np.pi, -np.inf, -np.inf, -np.inf, 0, 0, 0, 0]),
    high = np.array([np.inf, np.inf, np.inf, np.pi, np.pi, np.pi, np.inf, np.inf, np.inf, 1, 1, 1, 1]),
    shape=(13,),
    dtype=np.float32
)

env = AnymalEnv(
    robot = anymal,
    reward_type = 'none',
    n_obs = 12,
    goals = [100, 100, 0],
    action_space=action_space,
    observation_space=observation_space,
    random_start=False,
    max_steps=400
)

actor_net = Actor(observation_space.shape[0], np.prod(action_space.shape), action_lim)
critic_net = Critic(observation_space.shape[0], np.prod(action_space.shape))

train_td3(env, actor_net, critic_net)