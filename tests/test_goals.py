import pyfastsim as fastsim
import numpy as np

world_map = fastsim.Map('worlds/dots.pbm', 600)
disp = fastsim.Display(world_map, fastsim.Robot(10, fastsim.Posture(0, 0,0)))
goals = np.loadtxt('data/eval_data/dots.dat')
for i in goals:
    world_map.add_goal(fastsim.Goal(*i[:2], 10, 1))
while True:
    disp.update()