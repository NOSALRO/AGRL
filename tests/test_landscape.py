import numpy as np
import pickle
import torch
from nosalro.vae import StatesDataset, VariationalAutoencoder, train, visualize
from nosalro.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = Scaler()
with open('models/policies/1d/alley_mobile_edl_1/env.pickle', 'rb') as ef:
    env = pickle.load(ef)

vae = env.vae
scaler = env.scaler
dataset = env.goals
goals = []
# grid = np.random.uniform([0, 0, -np.pi], [600, 600, np.pi], size=(600, 3))
x = np.arange(0, 600, 12).reshape(-1,1)
y = np.arange(0, 600, 12).reshape(-1,1)
xy = np.hstack((x,y))
# theta = np.random.uniform(-np.pi, np.pi, xy.shape[0])
# print(len(np.arange(0, 1, 0.001)))
# print(grid)
x_hat, x_hat_var, latent, _ = vae(torch.tensor(dataset[0], device='cuda').float(), 'cuda', deterministic=True, scale=True)
dist = torch.distributions.MultivariateNormal(x_hat.squeeze().cpu(), torch.diag(x_hat_var.squeeze().exp().cpu()))
# xy = grid[:,:2]
# theta = grid[:,2]

heat = []
idx = 0
s = 2e+3
s = (0.85**0)*s
print(s)
for i in xy[:,0]:
    print(idx)
    for j in xy[:, 1]:
        _tmp = []
        reward = dist.log_prob(torch.tensor(scaler([i, j]))).cpu().item()
        # reward = np.linalg.norm(scaler([500, 500])-scaler([i, j]))
        _tmp.append(np.exp(reward/s))
        heat.append([i,j,np.mean(_tmp)])
    idx+=1

u = scaler(dataset[0], True)
fig, ax = plt.subplots()
ax.set_xlim((0, 600))
ax.set_ylim((0, 600))
circle1 = plt.Circle((u[0], u[1]), 3, color='r')
heat = np.array(heat)
sc = ax.scatter(heat[:,0], heat[:, 1], c=heat[:,2])
fig.colorbar(sc, shrink=0.8, aspect=5)
ax.add_patch(circle1)
plt.show()