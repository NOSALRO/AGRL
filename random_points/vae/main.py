import numpy as np
import torch
from vae_arch import VariationalEncoder, VAE, VariationalDecoder
import matplotlib.pyplot as plt

device = torch.device('cpu')
# datapoints = np.random.uniform(low=-1., high=1., size=(100000,2))
# np.save('random_data', datapoints)

class Data(torch.utils.data.Dataset):

    def __init__(self, path='random_data.npy'):

        # self.datapoints = np.load('random_data.npy')
        # self.datapoints = np.random.uniform(low=-1., high=1., size=(1000,2))
        self.datapoints = np.random.uniform(low=0., high=1., size=(10000,2))
        np.save('random_data', self.datapoints)

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        return self.datapoints[idx]

    def get_data(self):
        return self.datapoints

def loss_fn(x_target, x_hat, x_hat_var, mu, log_var, beta, eps=1e-06):
    gnll = torch.nn.functional.gaussian_nll_loss(x_hat, x_target, torch.exp(0.5 * x_hat_var))
    # mse = torch.nn.functional.mse_loss(x_hat, x_target)
    kld = beta * torch.nn.KLDivLoss(reduction='batchmean')(x_hat, x_target)
    # kld = beta * torch.mean(-0.5 * torch.sum(1. + log_var - mu**2 - log_var.exp(), dim=1))

    return gnll + kld

datapoints = Data()
dataset = torch.utils.data.DataLoader(dataset=datapoints, batch_size=128)

vae = VAE(2, 2, [128, 64]).to(device)
epochs = 200

for epoch in range(epochs):
    loss = []
    optimizer = torch.optim.Adam(vae.parameters(), lr=5e-5)
    for i in dataset:
        optimizer.zero_grad()
        x_hat, x_hat_var, mu, log_var = vae(i.float().to(device), device, False)
        _loss = loss_fn(i.float().to(device), x_hat, x_hat_var, mu, log_var, beta=0.1)
        _loss.backward()
        loss.append(_loss.item())
        optimizer.step()
    print(f"Epoch {epoch} -> loss = {np.mean(loss)}")


torch.save(vae, "../models/vae.pt")


x_hat, x_hat_var, mu, log_var = vae(torch.tensor(datapoints.get_data()).float(), 'cpu', True)
plt.scatter(x=datapoints[:,0], y=datapoints[:,1])
plt.scatter(x=x_hat.detach().numpy()[:,0], y=x_hat.detach().numpy()[:,1])
plt.show()