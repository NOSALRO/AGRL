import numpy as np
import torch
from nosalro.vae import StatesDataset, VariationalAutoencoder, train, visualize
from nosalro.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
scaler = Scaler()
target_scaler = Scaler(_type='standard')
shuffle = Shuffle(seed=42)

transforms = Compose([
    shuffle,
    # AngleToSinCos(angle_column=2),
    scaler.fit,
    scaler
])
dataset = StatesDataset(path='data/alley_go_explore.dat', transforms=transforms)

# In case output dims is different from input dims.
target_transforms = Compose([
    shuffle,
    target_scaler.fit,
    target_scaler
])
# target_dataset = StatesDataset(path='data/go_explore_1000.dat', transforms=target_transforms)

vae = VariationalAutoencoder(input_dims=2, latent_dims=2, output_dims=2, hidden_sizes=[64,64], scaler=scaler).to(device)
vae = train(
    model = vae,
    epochs = 500,
    lr = 1e-04,
    dataset = dataset,
    device = device,
    beta = 10,
    file_name = 'models/vae_models/alley_vae.pt',
    overwrite = False,
    weight_decay = 0,
    batch_size = 1024,
    # target_dataset = target_dataset
)
# i, x_hat_var, mu, logvar = vae(torch.tensor(dataset[:]).to(device), device, True, scale=False)
# visualize([dataset[:], i.detach().cpu().numpy()], projection='2d')


# grid = np.random.uniform([0, 0, -np.pi], [600, 600, np.pi], size=(600, 3))
x = np.arange(0, 600, 12).reshape(-1,1)
y = np.arange(0, 600, 12).reshape(-1,1)
xy = np.hstack((x,y))
# theta = np.random.uniform(-np.pi, np.pi, xy.shape[0])
# print(len(np.arange(0, 1, 0.001)))
# print(grid)
x_hat, x_hat_var, latent, _ = vae(torch.tensor(dataset[0], device='cuda').float(), 'cuda', deterministic=True, scale=False)
dist = torch.distributions.MultivariateNormal(x_hat.cpu(), torch.diag(x_hat_var.exp().cpu()))
# xy = grid[:,:2]
# theta = grid[:,2]

heat = []
idx = 0
for i in xy[:,0]:
    print(idx)
    for j in xy[:, 1]:
        _tmp = []
        reward = dist.log_prob(torch.tensor(scaler([i, j]))).cpu().item()
        _tmp.append(np.exp(reward/3.5e+3))
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