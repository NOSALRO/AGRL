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

# Create Graphics
gconfig = rd.gui.GraphicsConfiguration(1024, 768) # Create a window of 1024x768 resolution/size
graphics = rd.gui.Graphics(gconfig) # create graphics object with configuration
simu.set_graphics(graphics)
graphics.look_at([0., 3., 2.], [0., 0., 0.])

floor = simu.add_checkerboard_floor()
floor.set_draw_axis(floor.body_name(0), 1)
simu.step_world()

for n,i in enumerate(original_pose):
    rob = rd.Robot.create_ellipsoid(np.ones(3)/15, np.array(i, dtype=np.float32), "fixed", color=np.array([i[3], i[4] ,i[5], 0.8]), ellipsoid_name=str(n))
    if n == 10:
        rob.set_draw_axis(f'{n}', 1)
    simu.add_robot(rob)
    
while True:
    try:
        simu.step_world()
    except:
        break
