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
import numpy as np
import RobotDART as rd
import dartpy  # OSX breaks if this is imported before RobotDART
import copy

# Load a robot
robot = rd.Iiwa()
robot.set_actuator_types("velocity")

init_pos = [1., np.pi / 3., 0., -np.pi / 3., 0., np.pi / 3., 0.]
robot.set_positions(init_pos)

# Create simulator object
simu = rd.RobotDARTSimu()

# Create Graphics
gconfig = rd.gui.GraphicsConfiguration(1024, 768) # Create a window of 1024x768 resolution/size
graphics = rd.gui.Graphics(gconfig) # create graphics object with configuration
simu.set_graphics(graphics)
graphics.look_at([0., 3., 2.], [0., 0., 0.])

# Add robot and nice floor
simu.add_robot(robot)
simu.add_checkerboard_floor()

h = 0.2441
A = 0.16
period = 2. # in seconds
shift = 0.5 * period
B = 2. * np.pi / period
# _config.eef(0)->desired.pose.tail(3)[2] = A * std::cos(B * (x + shift)) + A + h;
# _config.eef(0)->desired.vel.tail(3)[2] = -B * A * std::sin(B * (x + shift));

t = 0.

rot_desired = robot.body_pose("iiwa_link_ee").rotation()
x_desired, y_desired = robot.body_pose_vec("iiwa_link_ee")[3:5]

positions = []
# Run simulation
for _ in range(200):
    des_p = A * np.cos(B* (t + shift)) + A + h
    des_v = -B * A * np.sin(B* (t + shift))

    x, y, z = robot.body_pose_vec("iiwa_link_ee")[3:]
    vz = robot.body_velocity("iiwa_link_ee")[5]

    rot = robot.body_pose("iiwa_link_ee").rotation()

    rot = 100. * rd.math.logMap(rot_desired @ rot.transpose())

    vel = 100. * (des_p - z) + 1. * (des_v - vz)

    all_vel = np.array([rot[0], rot[1], rot[2], 100. * (x_desired - x), 100. * (y_desired - y), vel]).reshape((6, 1))
    jac = robot.jacobian("iiwa_link_ee")
    jac_pinv = np.linalg.pinv(jac)
    dq = jac_pinv @ all_vel

    robot.set_commands(dq)
    if simu.step_world():
        break
    t += simu.timestep()

    print(robot.positions())
    positions.append(robot.positions())
    # print("Number of samples: ", len(positions))
# np.savetxt('data/eval_data/iiwa.dat', np.array(positions, dtype=np.float32))
