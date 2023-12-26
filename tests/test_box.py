from agrl.env import Box
import numpy as np

observation_space = Box(
    low=np.array([0, 0, -1, -1]),
    high=np.array([600, 600, 1, 1]),
    shape=(4,),
    dtype=np.float32
)
print(observation_space.scale(np.array([-1., 1.]), -1, 1))