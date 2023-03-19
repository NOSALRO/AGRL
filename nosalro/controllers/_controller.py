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
        rot_error = np.arctan2(xy_diff[1],xy_diff[0]) - current[-1]
        lin_error = np.linalg.norm(xy_diff)
        return np.r_[rot_error, lin_error]

    def update_rot(self, current):
        err = self.error(current)[0]
        self.__sum_rot_error += np.abs(err)
        _cmd = self.Kp*np.abs(err) + self.Ki*self.__sum_rot_error
        _cmd = np.clip(_cmd, a_min=0, a_max=.2)
        _cmd = [0, _cmd]
        _cmd = np.multiply(np.sign(err), _cmd)
        return  _cmd if np.abs(err) > 1e-5 else None

    def update_lin(self, current):
        err = self.error(current)[1]
        self.__sum_lin_error += err
        _cmd = self.Kp*err + self.Ki*self.__sum_lin_error
        _cmd = np.clip(_cmd, a_min=0, a_max=.3)
        _cmd = [_cmd, _cmd]
        return _cmd if err != 0 else None

    def update(self, current):
        cmd = self.update_rot(current)
        if cmd is None:
            cmd =  self.update_lin(current)
        if cmd is None:
            print('TRUE')
        return [0, 0] if cmd is None else cmd