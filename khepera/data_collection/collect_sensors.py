import sys
import time
import numpy as np
import pyfastsim as fastsim

def get_observations(robot):

    observations = np.array([])
    for laser in robot.get_lasers():
        observations = np.hstack([observations, laser.get_dist()])

    for laser_scanner in robot.get_laser_scanners():
        for laser in laser_scanner.get_lasers():
            observations = np.hstack([observations, laser.get_dist()])

    for radar in robot.get_radars():
        observations = np.hstack([observations, radar.get_activated_slice()])

    for light_sensor in robot.get_light_sensors():
        observations = np.hstack([observations, light_sensor.get_distance()])

    observations = np.hstack([observations, robot.get_camera().pixels()])

    return np.array(observations)

if __name__ == '__main__':

    map = fastsim.Map("worlds/three_wall.pbm", 600)
    map.add_goal(fastsim.Goal(100, 100, 10, 0))
    map.add_illuminated_switch(fastsim.IlluminatedSwitch(1, 10, 100, 240, True))
    # No need for goal in data collection phase.
    robot = fastsim.Robot(20., fastsim.Posture(100, 200, 0))

    # Sensor list.
    # robot.add_laser(fastsim.Laser(np.pi / 4.0, 50.))
    # robot.add_laser_scanner(fastsim.LaserScanner(-np.pi/4., np.pi/4., 0.01, 100.))
    # robot.add_radar(fastsim.Radar(0, 8, False))
    # robot.add_light_sensor(fastsim.LightSensor(1, np.pi/2., 100))
    # robot.use_camera(fastsim.LinearCamera(100., 600))

    d = fastsim.Display(map, robot)
    states = np.array([])
    for _ in range(1):
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
            # print(f"Step {i} robot pos: x = {x} y = {y} theta = {robot.get_pos().theta()}")
            if len(states) == 0:
                states = get_observations(robot)
            else:
                states = np.vstack([states, get_observations(robot)])
            robot.move(*moves, map, False)

    np.savetxt(sys.argv[1], states)