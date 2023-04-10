import numpy as np
import copy


class SingleRigidBodyDynamics:
    def __init__(self, mass, inertia, feet_ref_positions, feet_min_bounds, feet_max_bounds, T, T_swing):
        # # General state
        # self._mass
        # self._inertia
        # self._inertia_inv

        # # Global/Environment vars
        # self._dt
        # self._g
        # self._gravity

        # # COM state
        # self._base_position
        # self._base_vel
        # self._base_orientation
        # self._base_angular_vel # this is in local (COM) frame

        # # Feet state
        # self._feet_positions # in world frame, these are more of targets rather than states
        # self._feet_phases

        # # Static ref poses/bounds
        # self._feet_ref_positions # in COM frame
        # self._feet_min_bounds
        # self._feet_max_bounds

        # Sim
        self.set_sim_data(0.01, -9.81)

        # Body-related
        self.set_inertial_data(mass, inertia)

        # Feet-related
        self.set_feet_data(feet_ref_positions, feet_min_bounds, feet_max_bounds)

        # Zero-out entities
        self._base_position = np.zeros((3, 1))
        self._base_vel = np.zeros((3, 1))
        self._base_orientation = np.eye(3)
        self._base_angular_vel = np.zeros((3, 1))

        n_feet = len(self._feet_ref_positions)
        self._feet_positions = [np.zeros((3, 1))]*n_feet
        self._feet_phases = [0]*n_feet

        # Phases
        self._T = int(T)
        self._T_swing = int(T_swing)


    def set_sim_data(self, dt, gravity):
        self._dt = dt
        self._g = np.abs(gravity)
        self._gravity = np.array([[0.], [0.], [-self._g]])

    def set_inertial_data(self, mass, inertia):
        self._mass = mass
        self._inertia = inertia
        self._inertia_inv = np.linalg.inv(inertia)

    def set_feet_data(self, feet_ref_positions, feet_min_bounds, feet_max_bounds):
        self._feet_ref_positions = copy.deepcopy(feet_ref_positions)
        self._feet_min_bounds = copy.deepcopy(feet_min_bounds)
        self._feet_max_bounds = copy.deepcopy(feet_max_bounds)

    def set_data(self, base_position, base_velocity, base_orientation, base_angular_velocity, feet_positions, feet_phases):
        # Base-related
        self._base_position = copy.deepcopy(base_position)
        self._base_vel = copy.deepcopy(base_velocity)
        self._base_orientation = copy.deepcopy(base_orientation)
        self._base_angular_vel = copy.deepcopy(base_angular_velocity)

        # Feet-related
        self._feet_positions = copy.deepcopy(feet_positions)
        self._feet_phases = copy.deepcopy(feet_phases)

    def _skew(self, vec):
        skew = np.zeros((3, 3))
        skew[0, 1] = -vec[2]
        skew[0, 2] = vec[1]
        skew[1, 0] = vec[2]
        skew[1, 2] = -vec[0]
        skew[2, 0] = -vec[1]
        skew[2, 1] = vec[0]
        return skew

    def _exp_map(self, vec, epsilon = 1e-12):
        ret = np.eye(3)

        s2 = [vec[0] * vec[0], vec[1] * vec[1], vec[2] * vec[2]]
        s3 = vec[0] * vec[1], vec[1] * vec[2], vec[2] * vec[0]

        theta = np.sqrt(s2[0] + s2[1] + s2[2])
        cos_theta = np.cos(theta)
        alpha = 0.
        beta = 0.

        if theta > epsilon:
            alpha = np.sin(theta) /  theta
            beta = (1. - cos_theta) / theta / theta
        else:
            alpha = 1. - theta * theta / 6.
            beta = 0.5 - theta * theta /  24.

        ret[0, 0] = beta * s2[0] + cos_theta
        ret[1, 0] = beta * s3[0] + alpha * vec[2]
        ret[2, 0] = beta * s3[2] - alpha * vec[1]

        ret[0, 1] = beta * s3[0] - alpha * vec[2]
        ret[1, 1] = beta * s2[1] + cos_theta
        ret[2, 1] = beta * s3[1] + alpha * vec[0]

        ret[0, 2] = beta * s3[2] + alpha * vec[1]
        ret[1, 2] = beta * s3[1] - alpha * vec[0]
        ret[2, 2] = beta * s2[2] + cos_theta

        return ret

    def integrate(self, feet_forces, external_force = None, terrain_height = 0.):
        f_total = np.zeros((3, 1)) # without gravity
        for i in range(len(feet_forces)):
            is_swing = (self._feet_phases[i] % self._T) < self._T_swing
            if not is_swing:
                f_total += feet_forces[i]

        tau_total = np.zeros((3, 1))
        for i in range(len(feet_forces)):
            is_swing = (self._feet_phases[i] % self._T) < self._T_swing
            if not is_swing:
                tau_total += self._skew(self._feet_positions[i] - self._base_position) @ feet_forces[i]

        lin_acc = f_total / self._mass + self._gravity
        if external_force:
            lin_acc += external_force / self._mass
        ang_acc = self._inertia_inv @ (self._base_orientation.T @ tau_total - self._skew(self._base_angular_vel) @ self._inertia @ self._base_angular_vel)

        # semi-implicit Euler
        self._base_vel = self._base_vel + lin_acc * self._dt
        self._base_position = self._base_position + self._base_vel * self._dt

        self._base_angular_vel = self._base_angular_vel + ang_acc * self._dt
        self._base_orientation = self._base_orientation @ self._exp_map(self._base_angular_vel * self._dt)

        # phase counters + feet targets
        k_foot = 2. * (self._T - self._T_swing) * self._dt
        n_feet = len(self._feet_ref_positions)
        for i in range(n_feet):
            self._feet_phases[i] += 1 # increase phase variables
            if ((self._feet_phases[i] % self._T) == 0) and self._T_swing > 0:
                self._feet_positions[i][:2] = self._feet_ref_positions[i][:2] + k_foot * (self._base_orientation.T @ self._base_vel)[:2]
                self._feet_positions[i][2] = 0.

                self._feet_positions[i] = self._base_position + self._base_orientation @ self._feet_positions[i]
                self._feet_positions[i][2] = terrain_height

    def valid(self):
        n_feet = len(self._feet_ref_positions)
        for k in range(n_feet):
            is_swing = (self._feet_phases[k] % self._T) < self._T_swing
            if not is_swing:
                p = self._base_orientation.T @ (self._feet_positions[k] - self._base_position)
                for idx in range(3):
                    if self._feet_min_bounds[k][idx] > p[idx] or self._feet_max_bounds[k][idx] < p[idx]:
                        return False
        return True

def _inertia_tensor(self, Ixx, Iyy, Izz, Ixy = 0., Ixz = 0., Iyz = 0.):
    return np.array([[Ixx, -Ixy, -Ixz], [-Ixy, Iyy, -Iyz], [-Ixz, -Iyz, Izz]])

def create_anymal():
    inertia = _inertia_tensor(0.946438, 1.94478, 2.01835, 0.000938112, -0.00595386, -0.00146328)
    anymal_mass = 30.4213964625
    x_nominal_b = 0.34
    y_nominal_b = 0.19
    z_nominal_b = -0.42
    dx = 0.15
    dy = 0.1
    dz = 0.1

    T = 40
    T_swing = 20

    feet_ref_positions = []
    feet_min_bounds = []
    feet_max_bounds = []

    feet_positions = []

    # Forward Left
    feet_positions.append(np.array([x_nominal_b, y_nominal_b, 0.]).reshape((3, 1)))
    feet_min_bounds.append(np.array([x_nominal_b - dx, y_nominal_b - dy, z_nominal_b - dz]).reshape((3, 1)))
    feet_max_bounds.append(np.array([x_nominal_b + dx, y_nominal_b + dy, z_nominal_b + dz]).reshape((3, 1)))
    feet_ref_positions.append(np.array([x_nominal_b, y_nominal_b, z_nominal_b]).reshape((3, 1)))

    # Back Left
    feet_positions.append(np.array([-x_nominal_b, y_nominal_b, 0.]).reshape((3, 1)))
    feet_min_bounds.append(np.array([-x_nominal_b - dx, y_nominal_b - dy, z_nominal_b - dz]).reshape((3, 1)))
    feet_max_bounds.append(np.array([-x_nominal_b + dx, y_nominal_b + dy, z_nominal_b + dz]).reshape((3, 1)))
    feet_ref_positions.append(np.array([-x_nominal_b, y_nominal_b, z_nominal_b]).reshape((3, 1)))

    # Forward Right
    feet_positions.append(np.array([x_nominal_b, -y_nominal_b, 0.]).reshape((3, 1)))
    feet_min_bounds.append(np.array([x_nominal_b - dx, -y_nominal_b - dy, z_nominal_b - dz]).reshape((3, 1)))
    feet_max_bounds.append(np.array([x_nominal_b + dx, -y_nominal_b + dy, z_nominal_b + dz]).reshape((3, 1)))
    feet_ref_positions.append(np.array([x_nominal_b, -y_nominal_b, z_nominal_b]).reshape((3, 1)))

    # Back Right
    feet_positions.append(np.array([-x_nominal_b, -y_nominal_b, 0.]).reshape((3, 1)))
    feet_min_bounds.append(np.array([-x_nominal_b - dx, -y_nominal_b - dy, z_nominal_b - dz]).reshape((3, 1)))
    feet_max_bounds.append(np.array([-x_nominal_b + dx, -y_nominal_b + dy, z_nominal_b + dz]).reshape((3, 1)))
    feet_ref_positions.append(np.array([-x_nominal_b, -y_nominal_b, z_nominal_b]).reshape((3, 1)))

    robot = SingleRigidBodyDynamics(anymal_mass, inertia, feet_ref_positions, feet_min_bounds, feet_max_bounds, T, T_swing)

    robot._base_position = np.array([0., 0., np.abs(z_nominal_b)]).reshape((3, 1))
    robot._feet_positions = feet_positions
    robot._feet_phases = [T_swing, 0, 0, T_swing]

    return robot


# Simple simulation
anymal = create_anymal()

feet_forces = [np.array([[0.], [0.], [anymal._mass * np.abs(anymal._g)/2.]])]*4

for i in range(1001):
    print(i*anymal._dt)
    print(anymal._base_position.T)
    print(anymal._base_orientation)

    anymal.integrate(feet_forces)

    if not anymal.valid():
        print("Out of bounds")
        break
    print("================================")
