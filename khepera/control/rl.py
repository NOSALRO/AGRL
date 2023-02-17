import sys
import numpy as np
import torch
from gym_env import KheperaEnv
from utils import get_model_params
from stable_baselines3 import SAC, PPO

sys.path.append('../vae/')
from vae_arch import VAE, VariationalDecoder, VariationalEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
datapoints = np.loadtxt("../data_collection/datasets/three_wall.dat")
vae = torch.load("../models/vae_three_wall.pt")

target = datapoints[10000]
x_hat, x_hat_var, latent, _ = vae(torch.tensor(target, device=device).float(), device, True, True)
distribution = torch.distributions.MultivariateNormal(x_hat.cpu(), torch.diag(x_hat_var.mul(.5).exp().cpu()))

env = KheperaEnv(True, reward_dist=distribution, min_max_scale=[vae.min, vae.max], max_steps=1000, posture=[100,50,0])
env.render()
env.set_target(target)
policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                    net_arch=dict(pi=[16, 16], qf=[16, 16]))

# model = SAC('MlpPolicy', env, 5e-4, verbose=1, policy_kwargs=policy_kwargs)
# model = SAC('MlpPolicy', env, 1e-4, verbose=1, batch_size=256)
model = SAC('MlpPolicy', env, 1e-4, verbose=1, batch_size=256, policy_kwargs=dict(activation_fn=torch.nn.Tanh))
model.learn(total_timesteps=100000)
