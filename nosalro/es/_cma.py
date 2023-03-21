import torch
import copy
import cma
from ._utils import set_model_params, get_model_params

def env_run(params, env, net, *, device='cpu', graphics=False):
    _env = copy.deepcopy(env)
    _net = copy.deepcopy(net)
    set_model_params(_net, params, device=device)
    reward = 0
    obs = _env.reset()
    if graphics:
        _env.render()
    action = _net(torch.tensor(obs, device=device))
    done = False
    while not done:
        obs, reward, done, _ = _env.step(action.cpu().detach().numpy())
        action = _net(torch.tensor(obs, device=device))
    return -reward

def cma_run(env, net):
    best_solution = cma.evolution_strategy.fmin(env_run, x0=get_model_params(net), sigma0=0.5, options={'CMA_elitist' : True, "maxfevals" : 5}, args=(env, net))
    return best_solution

def cma_eval(env, net, best_solution):
    set_model_params(net, best_solution[0], device='cpu')
    torch.save(net, 'models/policies/cma_policy.pt')
    env_run(best_solution[0], env, net, graphics=True)
