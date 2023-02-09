import copy
import numpy as np
import torch


def run(params, robot, net, distribution, target, verbose=0):
    _robot = copy.deepcopy(robot)
    _net = copy.deepcopy(net)
    set_model_params(_net, params)
    _reward = 0
    for _ in range(200):
        state = torch.tensor(_robot.state()).float()
        action = _net(state)
        _robot.step(action.detach().numpy())
        _reward += reward(torch.tensor(_robot.state()), distribution)
        if verbose == 2:
            print(f'Robot Pos: {state.numpy()}, Target Pos: {target}, Action: {action.detach().numpy()}')
    if verbose:
        print(f'Reward: {_reward} | Robot Pos: {_robot.state()}, Target Pos: {target}')
    return -_reward

def get_model_params(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).detach().numpy()

def set_model_params(model, params):
    torch.nn.utils.vector_to_parameters(torch.Tensor(params), model.parameters())

def reward(state, distribution):
    return distribution.log_prob(state).detach().item()