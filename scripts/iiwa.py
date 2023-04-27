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

# Run simulation
while True:
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

    print(robot.body_pose_vec("iiwa_link_ee")[3:])
