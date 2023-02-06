import numpy as np
from scipy.spatial.transform import Rotation as R
import scipy.special

class Robot:

    def __init__(self, mass, inertia):
        self.position = np.zeros((3,1), dtype=np.float64)
        self.orientation = np.eye(3, dtype=np.float64)
        self.velocity = np.zeros((3,1), dtype=np.float64)
        self.angular_velocity = np.zeros((3,1), np.float64)
        self.mass = mass
        self.inertia = self.__inertia_to_matrix(inertia)
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
        self.orientation *= self.__exp_mapping(__angular_accel*dt, self.orientation)

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

    @staticmethod
    def __exp_mapping(v, R, epsilon = 1e-12):
        v = v.reshape(1,-1)[0]
        R = R[0]
        s2 = v**2
        s3 = [v[0]*v[1], v[1]*v[2], v[2]*v[0]]
        theta = np.sqrt(np.sum(s2))
        cos_theta = np.cos(theta)
        alpha = 0
        beta = 0

        if theta > epsilon:
            alpha = np.sin(theta)/theta
            beta = (1 - cos_theta) / theta**2
        else:
            alpha = 1 - theta**2/6.
            beta = 0.5 - theta**2/24.

        R[0][0] = beta * s2[0] + cos_theta
        R[1][0] = beta * s3[0] + alpha * v[2]
        R[2][0] = beta * s3[2] - alpha * v[1]

        R[0][1] = beta * s3[0] - alpha * v[2]
        R[1][1] = beta * s2[1] + cos_theta
        R[2][1] = beta * s3[1] + alpha * v[0]

        R[0][2] = beta * s3[2] + alpha * v[1]
        R[1][2] = beta * s3[1] - alpha * v[0]
        R[2][2] = beta * s2[2] + cos_theta

        return R

    @staticmethod
    def __inertia_to_matrix(inertia):
        # correct order Ixx, Iyy, Izz, Ixy, Ixz, Iyz
        I = np.zeros((3,3))

        I[0][0] = inertia[0]
        I[0][1] = -inertia[3]
        I[0][2] = -inertia[4]

        I[1][0] = -inertia[3]
        I[1][1] = inertia[1]
        I[1][2] = -inertia[5]

        I[2][0] = -inertia[4]
        I[2][1] = -inertia[5]
        I[2][2] = inertia[2]

        return I