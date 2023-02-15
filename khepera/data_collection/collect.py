import sys, math
import numpy as np
import pyfastsim as fastsim
import time

if __name__ == '__main__':

    map = fastsim.Map("worlds/three_wall.pbm", 600)
    # No need for goal in data collection phase.
    # map.add_goal(fastsim.Goal(100, 100, 10, 0))
    robot = fastsim.Robot(20., fastsim.Posture(100, 200, 0))
    robot.add_laser(fastsim.Laser(math.pi / 4.0, 100.0))
    robot.add_laser(fastsim.Laser(-math.pi / 4.0, 100.0))
    robot.add_laser(fastsim.Laser(0., 100.))
    # Radar needs goal to work.
    # robot.add_radar(fastsim.Radar(0, 4))
    d = fastsim.Display(map, robot)
    states = []
    for _ in range(20):
        _states = []
        x_pos, y_pos = np.random.uniform(low=20, high=550, size=2)
        pixel_type = str(map.get_real(float(x_pos), float(y_pos)))
        while pixel_type == 'status_t.obstacle':
            x_pos, y_pos = np.random.uniform(low=0, high=600, size=2)
            pixel_type = str(map.get_real(float(x_pos), float(y_pos)))
        robot.set_pos(fastsim.Posture(x_pos, y_pos, 0))
        for i in range(2000):
            d.update()
            if not (i % 100):
                moves = np.random.uniform(low=-1, high=1, size=2)
            x, y = robot.get_pos().x(), robot.get_pos().y()
            print(f"Step {i} robot pos: x = {x} y = {y} theta = {robot.get_pos().theta()}")
            _states.append([x, y, robot.get_pos().theta()])
            robot.move(moves[0], moves[1], map, False)
        # Do not keep if stuck.
        _, indices = np.unique(np.array(_states)[:, :2], axis=0, return_index=True)
        if len(indices)>1:
            states.append(_states)

    states = np.vstack(states)
    np.savetxt(sys.argv[1], states)