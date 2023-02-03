import numpy as np
from scipy.spatial.transform import Rotation as R

class Robot:

    def __init__(self, mass, inertia):
        self.position = np.zeros((3,1), dtype=np.float64)
        self.orientation = np.zeros((3,3), dtype=np.float64)
        self.velocity = np.zeros((3,1), dtype=np.float64)
        self.angular_velocity = np.zeros((3,1), np.float64)
        self.mass = mass
        self.inertia = inertia
        self.inertia_inv = np.linalg.inv(self.inertia)

    def set_initial_position(self, postion):
        self.position = postion

    def set_initial_orientation(self, orientation):
        self.orientation = R.from_euler('zyx', orientation.reshape(1,-1)).as_matrix()

    def get_position(self):
        return self.position

    def get_orientation(self):
        return R.from_matrix(self.orientation).as_euler('zyx')

    def step(self, force, torque, dt):

        __accel = force/self.mass
        __angular_accel = self.inertia_inv @ ((self.orientation.T.reshape(3,3).dot(torque)) - (self.__skew_symetric(self.angular_velocity) @ self.inertia @ self.angular_velocity))

        self.velocity += __accel*dt
        self.position += self.velocity*dt
        self.angular_velocity += __angular_accel*dt
        self.orientation *= np.exp(self.__skew_symetric(self.angular_velocity*dt))

    def state(self):
        return np.vstack((
            self.position.reshape(-1, 1),
            self.velocity.reshape(-1, 1),
            self.get_orientation().reshape(-1,1),
            self.angular_velocity.reshape(-1, 1)
        ))


    def __skew_symetric(self, vel):

        skew_sym = [
            [0., -vel[2][0], vel[1][0]],
            [vel[2][0], 0., -vel[0][0]],
            [-vel[1][0], vel[0][0], 0.]
        ]
        return np.array(skew_sym, dtype=np.float32)
