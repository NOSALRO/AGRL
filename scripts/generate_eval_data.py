import argparse
import torch
import numpy as np
import pyfastsim as fastsim


def generate_points(N, world_map, file_name):
    world_map = fastsim.Map(world_map, 600)
    goals = []
    while len(goals) < N:
        point = np.random.uniform(0, 600, 2)
        if not fastsim.Robot(10, fastsim.Posture(*point,0))._check_collision(world_map):
            goals.append(point)
    goals = np.array(goals)
    if file_name is not None:
        np.savetxt(file_name, goals)
    return goals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type= int, help="Number of generated points.")
    parser.add_argument("--map", help="map file path. Choose between train, continue and eval mode.")
    parser.add_argument("--file-name", default=None, help="file name to save the data.")
    parser.add_argument("-g", "--graphics", action='store_true', help="enable graphics to preview the targets.")
    args = parser.parse_args()
    goals = generate_points(args.N, args.map, args.file_name)
    world_map = fastsim.Map(args.map, 600)
    if args.graphics:
        disp = fastsim.Display(world_map, fastsim.Robot(10, fastsim.Posture(0, 0,0)))
        for i in goals:
            world_map.add_goal(fastsim.Goal(*i[:2], 10, 1))
        while True:
            disp.update()