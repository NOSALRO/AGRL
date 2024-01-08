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
# import pyfastsim as fastsim
# import numpy as np

# world_map = fastsim.Map('worlds/dots.pbm', 600)
# disp = fastsim.Display(world_map, fastsim.Robot(10, fastsim.Posture(0, 0,0)))
# goals = np.loadtxt('data/go_explore_xy.dat')
# for i in goals:
#     world_map.add_goal(fastsim.Goal(*i[:2], 10, 1))
# while True:
#     disp.update()



import numpy as np
import RobotDART as rd
import dartpy  # OSX breaks if this is imported before RobotDART
import copy

simu = rd.RobotDARTSimu()


robot = rd.Iiwa()
original_pose = []
for i in np.loadtxt('data/iiwa_ge.dat'):
    robot.set_positions([*i])
    original_pose.append(robot.body_pose_vec("iiwa_link_ee"))

robot.set_actuator_types("velocity")

init_pos = [1., 8.30431278e-01, -5.26155741e-12, -7.25554906e-01, 5.56387497e-12 , 1.58158558e+00, -2.10255835e-12]
robot.set_positions(init_pos)
# Create Graphics
gconfig = rd.gui.GraphicsConfiguration(1024, 768) # Create a window of 1024x768 resolution/size
graphics = rd.gui.Graphics(gconfig) # create graphics object with configuration
simu.set_graphics(graphics)
graphics.look_at([0., 3., 2.], [0., 0., 0.])

floor = simu.add_checkerboard_floor()
floor.set_draw_axis(floor.body_name(0), 1)
simu.step_world()

for n,i in enumerate(original_pose):
    rob = rd.Robot.create_ellipsoid(np.ones(3)/15, np.array(i, dtype=np.float32), "fixed", color=np.array([0, 128 , 128, 0.05]), ellipsoid_name=str(n))
    rob.set_cast_shadows(False)
    simu.add_robot(rob)

simu.add_robot(robot)

while True:
    try:
        simu.step_world()
    except:
        break
