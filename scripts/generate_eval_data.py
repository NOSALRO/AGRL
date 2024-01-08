#!/usr/bin/env python
# encoding: utf-8
#|
#|    Copyright (c) 2022-2023 Computational Intelligence Lab, University of Patras, Greece
#|    Copyright (c) 2022-2023 Constantinos Tsakonas
#|    Copyright (c) 2022-2023 Konstantinos Chatzilygeroudis
#|    Authors:  Constantinos Tsakonas
#|              Konstantinos Chatzilygeroudis
#|    email:    tsakonas.constantinos@gmail.com
#|              costashatz@gmail.com
#|    website:  https://nosalro.github.io/
#|              http://cilab.math.upatras.gr/
#|
#|    This file is part of AGRL.
#|
#|    All rights reserved.
#|
#|    Redistribution and use in source and binary forms, with or without
#|    modification, are permitted provided that the following conditions are met:
#|
#|    1. Redistributions of source code must retain the above copyright notice, this
#|       list of conditions and the following disclaimer.
#|
#|    2. Redistributions in binary form must reproduce the above copyright notice,
#|       this list of conditions and the following disclaimer in the documentation
#|       and/or other materials provided with the distribution.
#|
#|    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#|    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#|    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#|    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#|    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#|    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#|    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#|    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#|    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#|    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#|
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