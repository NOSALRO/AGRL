import numpy as np


class Controller:
    def __init__(self, target, Kp, Ki):
        self._target = target
        self.Kp = Kp
        self.Ki = Ki
        self.__sum_rot_error = 0
        self.__sum_lin_error = 0

    def set_target(self, target):
        self._target = target

    def reset(self):
        self.__sum_rot_error = 0
        self.__sum_lin_error = 0

    def error(self, current):
        xy_diff = np.array(self._target) - np.array(current[:-1])
        rot_error = np.arctan2(xy_diff[1], xy_diff[0]) - current[-1]
        lin_error = np.linalg.norm(xy_diff)
        return np.r_[rot_error, lin_error]

    def update(self, current):
        rot_error, lin_error = self.error(current)
        self.__sum_rot_error += np.abs(rot_error)
        self.__sum_lin_error += lin_error

        _rot_cmd = self.Kp*np.abs(rot_error) + self.Ki*self.__sum_rot_error
        _rot_cmd = np.clip(_rot_cmd, a_min=0., a_max=0.3)
        _rot_cmd = [0, _rot_cmd]
        _rot_cmd = np.multiply(np.sign(rot_error), _rot_cmd)

        if np.abs(rot_error) < 1e-01:
            lin_err = self.error(current)[1]
            _cmd = self.Kp*lin_error + self.Ki*self.__sum_lin_error
            _cmd = np.clip(_cmd, a_min=0., a_max=0.5)
            _cmd = [_cmd, _cmd] + _rot_cmd
            return _cmd if [lin_error, np.abs(rot_error)] > [1e-02, 1e-02] else None
        return _rot_cmd