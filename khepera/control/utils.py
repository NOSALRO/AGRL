import copy, math
import numpy as np
import torch
import pyfastsim as fastsim


def run(params, net, distribution, target, m, _min, _max, verbose=0):
    _robot = fastsim.Robot(20., fastsim.Posture(200, 250, 0))
    _robot.add_laser(fastsim.Laser(math.pi / 4.0, 100.0))
    _robot.add_laser(fastsim.Laser(-math.pi / 4.0, 100.0))
    _robot.add_laser(fastsim.Laser(0., 100.))
    if verbose == 2:
        d = fastsim.Display(m, _robot)
    _net = copy.deepcopy(net)
    set_model_params(_net, params)
    _reward = 0
    steps = 5000 if verbose else 600
    for i in range(steps):
        state = torch.tensor([_robot.get_pos().x(), _robot.get_pos().y(), _robot.get_pos().theta()]).float()
        if verbose != 2:
            action = _net(state).detach().numpy() + np.random.normal(loc=0, scale=0.5, size=2)
        else:
            action = _net(state).detach().numpy()
        if verbose == 2:
            d.update()
        _robot.move(action[0], action[1], m, False)
        current_state = [_robot.get_pos().x(), _robot.get_pos().y(), _robot.get_pos().theta()]
        current_state = torch.tensor((current_state - _min)/(_max - _min))
        _reward += reward(current_state.cpu(), distribution)
        if verbose == 2:
            print(f'Robot Pos: {state}, Target Pos: {target}, Action: {action}')
    if verbose:
        print(f'Reward: {_reward} | Robot Pos: {[_robot.get_pos().x(), _robot.get_pos().y(), _robot.get_pos().theta()]}, Target Pos: {target}')
    return -_reward

def get_model_params(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).cpu().detach().numpy()

def set_model_params(model, params):
    torch.nn.utils.vector_to_parameters(torch.Tensor(params), model.parameters())

def reward(state, distribution):
    return distribution.log_prob(state).detach().cpu().item()