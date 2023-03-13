import numpy as np


class Controller:
    def __init__(self, target):
        self._target = target

    def set_target(self, target):
        self._target = target

    def error(self, current):
        xy_diff = np.array(self._target) - np.array(current[:-1])
        rot_error = np.arctan2(xy_diff[1],xy_diff[0]) - current[-1]
        lin_error = np.linalg.norm(xy_diff, ord=2)
        return np.r_[rot_error, lin_error]

    def update_rot(self, current):
        err = self.error(current)[0]
        _cmd = np.clip([0, np.abs(err) * .25], a_min=0, a_max=1)
        return  np.multiply(np.sign(err), _cmd) if np.abs(err) > 1e-05 else None

    def update_lin(self, current):
        err = self.error(current)[1]
        Kp = 0.01
        _cmd = np.clip([err * Kp, err * Kp], a_min=-1, a_max=1)
        return _cmd if err > 1e-05 else None

    def update(self, current):
        cmd = self.update_rot(current)
        if cmd is None:
            cmd =  self.update_lin(current)
        return [0, 0] if cmd is None else cmd