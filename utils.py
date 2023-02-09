import matplotlib.pyplot as plt
import numpy as np
import torch

def vizualization(data, color=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    c = color
    if color is None:
        c = data[:,2]
    state_space = ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=c, cmap='coolwarm')
    fig.colorbar(state_space, shrink=0.8, aspect=5)
    plt.show()

def plot_latent_reconstructed(model, N, device):
    latent_vectors = []
    latent_vectors, states, _ = model.sample(N, device)
    vizualization(latent_vectors)
    vizualization(states.cpu().detach())

# def loss_fn(x_target, x_hat, x_hat_var, mean, log_var, beta = 1e-4):
#     max_logvar = 0.5
#     min_logvar = -10
#     log_v = max_logvar - ((max_logvar - x_hat_var).exp() + 1.).log()
#     log_v = min_logvar + ((log_v - min_logvar).exp() + 1.).log()
#     var = x_hat_var.exp()
#     mse = torch.nn.functional.gaussian_nll_loss(x_hat, x_target, var, full=True)
#     kl_divergence = beta * torch.mean(-0.5 * torch.sum(1. + log_var - mean.pow(2) - log_var.exp(), dim=1))
#     return mse + kl_divergence

def loss_fn(x_target, x_hat, x_hat_var, mu, log_var, beta = 1e-4):
    gnll = torch.nn.functional.gaussian_nll_loss(x_hat, x_target, torch.exp(x_hat_var), full=True)
    kld = beta * torch.mean(-0.5 * torch.sum(1. + log_var - mu**2 - log_var.exp(), dim=1))
    return gnll + kld

def train(model, epochs, lr, dataloader, device, beta):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        loss = []
        for x in dataloader:
            optimizer.zero_grad()
            x = x.to(device)
            x_hat, x_hat_var, mu, log_var = model(x, device)
            _loss = loss_fn(x, x_hat, x_hat_var, mu, log_var, beta)
            _loss.backward()
            loss.append(_loss.cpu().detach())
            optimizer.step()
        print(f"Epoch: {epoch}, reconstruction + kl loss = {np.mean(loss)}")
    return model
