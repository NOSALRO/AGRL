import matplotlib.pyplot as plt
import numpy as np
import torch

def vizualization(data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    state_space = ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=data[:,2], cmap='coolwarm')
    fig.colorbar(state_space, shrink=0.8, aspect=5)
    plt.show()

def plot_latent_reconstructed(model, device):
    latent_vectors = []
    latent_vectors, states = model.sample(100, device)
    vizualization(latent_vectors)
    vizualization(states.cpu().detach())

def loss_fn(x_target, x_hat, mean, log_var):
    mse = torch.nn.MSELoss()(x_hat, x_target)
    kl_divergence = torch.mean(-0.5 * torch.sum(1 + log_var - mean**2  - log_var.exp(), dim=1))
    return mse + kl_divergence

def train(model, epochs, dataloader, device):

    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        l = []
        for x in dataloader:

            optimizer.zero_grad()
            x = x.to(device)
            x_hat, mean, log_var = model(x)
            loss = loss_fn(x, x_hat, mean, log_var)
            loss.backward()
            l.append(loss.cpu().detach())
            optimizer.step()

        print(f"Epoch: {epoch}, reconstruction + kl loss = {np.mean(l)}")
    return model
