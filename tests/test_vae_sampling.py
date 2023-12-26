import torch
import pickle
import numpy as np
from agrl.env import KheperaDVControllerEnv, Box
from agrl.controllers import DVController
from agrl.rl.td3 import Actor, Critic, train_td3
from agrl.vae import StatesDataset, VariationalAutoencoder, train, visualize
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import pyfastsim as fastsim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = Scaler()
with open('.tmp/alley_mobile_edl_1/env.pickle', 'rb') as ef:
    env = pickle.load(ef)

vae = env.vae
scaler = env.scaler
dataset = env.goals
goals = []

x_hat, x_hat_var, mu, log_var = vae(torch.tensor(dataset[:]).to(device), device, True, scale=True)
print(log_var.exp().detach().cpu().numpy().mean())
# for x in dataset:
#     for _ in range(10):
#         i, x_hat_var, mu, log_var = vae(torch.tensor(x).to(device), device, True, scale=True)
#         _, out, _ = vae.sample(1, device, mu.cpu(), log_var.mul(0.5).exp().cpu())
#         goals.append(scaler(out.cpu().detach().squeeze().numpy(), undo=True))
# visualize([np.array(goals)], projection='2d', file_name='.tmp/image_ge')
# visualize([np.array(goals)], projection='2d', file_name='.tmp/image_uniform')
# world_map = fastsim.Map('worlds/alley_right.pbm', 600)
# disp = fastsim.Display(world_map, fastsim.Robot(10, fastsim.Posture(0, 0,0)))
# for i in goals:
#     world_map.add_goal(fastsim.Goal(*i[:2], 10, 1))
# while True:
#     disp.update()

# i, x_hat_var, mu, logvar = vae(torch.tensor(dataset[:]).to(device), device, True, scale=False)
# visualize([dataset[:], i.detach().cpu().numpy()], projection='2d', file_name='.tmp/image')
