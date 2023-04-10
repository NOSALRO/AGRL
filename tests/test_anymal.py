import numpy as np
from scipy.spatial.transform import Rotation
from nosalro.robots import Anymal
from nosalro.env.anymal import AnymalEnv
from nosalro.env import Box


anymal = Anymal()
action_lim = 2*anymal._mass
action_space = Box(
    low=np.zeros((4,3,1)),
    high=np.multiply(np.ones((4,3,1)), action_lim),
    shape=(4,3,1),
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

feet_forces = [np.array([[0.], [0.], [anymal._mass * np.abs(anymal._g)/2.]])]*4
for i in range(10000):

    obs, reward, done, _ = env.step(feet_forces)
    if done:
        print(obs[:3])
        print("Out of bounds")
        print(i)
        print("================================")
