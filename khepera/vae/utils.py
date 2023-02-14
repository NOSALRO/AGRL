import matplotlib.pyplot as plt
import numpy as np
import torch

def vizualization(data, color=None, title=None):
    fig = plt.figure()
    ax = plt.axes()
    c = color
    if color is None:
        c = data[:,2]
    state_space = ax.scatter(data[:, 0], data[:, 1], cmap='coolwarm')
    fig.colorbar(state_space, shrink=0.8, aspect=5)
    ax.set_title(title)
    plt.show()

def plot_latent_reconstructed(model, N, device):
    latent_vectors = []
    latent_vectors, states, _ = model.sample(N, device)
    vizualization(latent_vectors, title="Sampled Data Latent Vectors")
    vizualization(states.cpu().detach(), title="Reconstructed Sampled Data")

def loss_fn(x_target, x_hat, x_hat_var, mu, log_var, beta, eps=1e-06):
    gnll = torch.nn.functional.gaussian_nll_loss(x_hat, x_target, torch.exp(0.5 * x_hat_var))
    kld = beta * torch.mean(-0.5 * torch.sum(1. + log_var - mu**2 - torch.exp(.5 * log_var), dim=1))
    return gnll + kld

def train(model, epochs, lr, dataloader, device, beta = 1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        l = []
        for x in dataloader:
            optimizer.zero_grad()
            x = x.to(device)
            x_hat, x_hat_var, mean, log_var = model(x, device)
            loss = loss_fn(x, x_hat, x_hat_var, mean, log_var, beta)
            loss.backward()
            l.append(loss.cpu().detach())
            optimizer.step()
        print(f"Epoch: {epoch}, reconstruction + kl loss = {np.mean(l)}")
    return model