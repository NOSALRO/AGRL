import sys, math
import numpy as np
import pyfastsim as fastsim

if __name__ == '__main__':

    map = fastsim.Map("worlds/three_wall.pbm", 600)
    # No need for goal in data collection phase.
    # map.add_goal(fastsim.Goal(100, 100, 10, 0))
    robot = fastsim.Robot(20., fastsim.Posture(100, 100, 0))
    robot.add_laser(fastsim.Laser(math.pi / 4.0, 100.0))
    robot.add_laser(fastsim.Laser(-math.pi / 4.0, 100.0))
    robot.add_laser(fastsim.Laser(0., 100.))
    # Radar needs goal to work.
    # robot.add_radar(fastsim.Radar(0, 4))
    d = fastsim.Display(map, robot)
    states = []
    for i in range(10000):
        d.update()
        if not (i % 100):
            moves = np.random.uniform(low=-1, high=1, size=2)
        print(f"Step {i} robot pos: x = {robot.get_pos().x()} y = {robot.get_pos().y()} theta = {robot.get_pos().theta()}")
        states.append([robot.get_pos().x(), robot.get_pos().y(), robot.get_pos().theta()])
        robot.move(moves[0], moves[1], map)
    states = np.array(states)
    np.savetxt(sys.argv[1], states)