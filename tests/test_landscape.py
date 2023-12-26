import numpy as np
import pickle
import torch
from agrl.vae import StatesDataset, VariationalAutoencoder, train, visualize
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('models/policies/2d/dots_mobile_edl_3/env.pickle', 'rb') as ef:
    env = pickle.load(ef)

vae = env.vae
scaler = env.scaler
dataset = env.goals
goals = []

# x = np.arange(0, 600, 12).reshape(-1,1)
# y = np.arange(0, 600, 12).reshape(-1,1)
# xy = np.hstack((x,y))
# theta = np.random.uniform(-np.pi, np.pi, xy.shape[0])

x = np.linspace(0, 600, 600)
y = np.linspace(0, 600, 600)
xy = []
for i in x:
    for j in y:
        xy.append([i,j])
xy = np.array(xy)
# print(len(np.arange(0, 1, 0.001)))
# print(grid)
s = 1e+5
# xy = grid[:,:2]
# theta = grid[:,2]
def logsumexp(x):
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

for i, d in enumerate([[350, 150]]):
    x_hat, x_hat_var, latent, _ = vae(torch.tensor(d , device='cuda').float(), 'cuda', deterministic=True, scale=True)
    dist = torch.distributions.MultivariateNormal(x_hat.cpu().squeeze(), torch.diag(x_hat_var.squeeze().mul(0.5).exp().cpu()))
    loc = scaler(x_hat.cpu().detach().squeeze().numpy(), True)
    print("X: ", loc[0], ", Y: ", loc[1])
    robot_pos = [loc[0], loc[1]]
    # robot_pos = [500,100]
    reward = dist.log_prob(torch.tensor(scaler([robot_pos[0], robot_pos[1]]))).cpu().detach().numpy()
    max_reward = dist.log_prob(torch.tensor(scaler([loc[0], loc[1]]))).cpu().detach().numpy()
    print("Max reward: ", max_reward)
    print("Reward: ", reward)
    print("Probability: ", np.exp(reward))
    heat_map = []
    max_r = 0
    rewards = dist.log_prob(torch.tensor(scaler(xy))).cpu().detach().numpy()
    reg = dist.log_prob(torch.tensor(scaler(dataset[:]))).cpu().detach().numpy()
    rewards = (rewards - reg.min())/(reg.max() - reg.min())
    rewards = rewards.reshape(xy.shape[0],1)
    heat_map = np.concatenate((xy, rewards),axis=1)
    # for robot_pos in xy:
    #     reward = dist.log_prob(torch.tensor(scaler([robot_pos[0], robot_pos[1]]))).cpu().detach().numpy()
    #     if reward != max_reward:
    #         heat_map.append([robot_pos[0], robot_pos[1], reward])

    heat_map = np.array(heat_map)
    fig, ax = plt.subplots()
    m = ax.scatter(heat_map[:,0], heat_map[:, 1], c=np.exp(heat_map[:,2]/(0.75*(0.85**6))))
    fig.colorbar(m, shrink=0.8, aspect=5)
    circle3 = plt.Circle(tuple(d), 3, color='white')
    circle2 = plt.Circle(tuple(xy[np.where(heat_map[:,2] == np.max(heat_map[:,2]))[0].item()]), 3, color='b')
    circle1 = plt.Circle((loc[0], loc[1]), 3, color='r')
    circle2 = plt.Circle(tuple(xy[np.where(heat_map[:,2] == np.max(heat_map[:,2]))[0].item()]), 3, color='b')
    ax.add_patch(circle1)
    ax.add_patch(circle2)
    ax.add_patch(circle3)
    # plt.show()
    plt.savefig(f'.tmp/landscape/im{i}.png')
# print("DONE")
# print("Max of targets: ", np.max(heat_map[:,2]))
# print(np.where(heat_map[:,2] == np.max(heat_map[:,2])))
# fig.colorbar(sc, shrink=0.8, aspect=5)
# for xh, xhv, k in zip(x_hat, x_hat_var, range(len(dataset))):
#     dist = torch.distributions.MultivariateNormal(xh.cpu().squeeze(), torch.diag(xhv.squeeze().exp().cpu()))
#     for p in range(k, k+1):
#         heat = []
#         idx = 0
#         for i in xy[:,0]:
#             print(idx)
#             for j in xy[:, 1]:
#                 _tmp = []
#                 reward = dist.log_prob(torch.tensor(scaler([i, j]))).cpu().item()
#                 # reward = np.linalg.norm(scaler([500, 500])-scaler([i, j]))
#                 _tmp.append(reward/1000)
#                 heat.append([i,j,np.mean(_tmp)])
#             idx+=1

#         u = scaler(dataset[0], True)
#         fig, ax = plt.subplots()
#         ax.set_xlim((0, 600))
#         ax.set_ylim((0, 600))
#         circle1 = plt.Circle((u[0], u[1]), 3, color='r')
#         heat = np.array(heat)
#         sc = ax.scatter(heat[:,0], heat[:, 1], c=heat[:,2])
#         fig.colorbar(sc, shrink=0.8, aspect=5)
        # ax.add_patch(circle1)
#         plt.savefig(f'.tmp/landscape/im{p}.png')