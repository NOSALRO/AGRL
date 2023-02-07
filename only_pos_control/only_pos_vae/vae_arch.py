import torch
import torch.nn.functional as F
import numpy as np

class VarEncoder(torch.nn.Module):
    def __init__(self, input_shape, latent_dims, hidden_sizes = [10, 10]) -> None:
        super().__init__()

        self.l1 = torch.nn.Linear(input_shape, hidden_sizes[0])
        self.l2 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = torch.nn.Linear(hidden_sizes[1], latent_dims)
        self.l4 = torch.nn.Linear(hidden_sizes[1], latent_dims)


    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        mu = self.l3(x)
        log_var = self.l4(x)

        return mu, log_var

class Decoder(torch.nn.Module):
    def __init__(self, input_shape, latent_dims, hidden_sizes = [10, 10]) -> None:
        super().__init__()

        self.l1 = torch.nn.Linear(latent_dims, hidden_sizes[0])
        self.l2 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = torch.nn.Linear(hidden_sizes[1], input_shape)
        self.l4 = torch.nn.Linear(hidden_sizes[1], input_shape)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        mu = self.l3(x)
        log_var = self.l4(x)

        return mu, log_var

class VarAutoencoder(torch.nn.Module):
    def __init__(self, input_shape, latent_dims, hidden_sizes = [10, 10]) -> None:
        super().__init__()

        self.latent_dims = latent_dims
        self.encoder = VarEncoder(input_shape, latent_dims, hidden_sizes)
        self.decoder = Decoder(input_shape, latent_dims, hidden_sizes[::-1])

    # Reparameterization trick to backprop stochastic node. σ * ε + μ
    def reparam(self, mean, var):
        eps = torch.randn_like(var)
        z = mean + eps * var
        return z

    def forward(self, x, deterministic = False):
        mean, log_var = self.encoder(x)
        z = mean if deterministic else self.reparam(mean, torch.exp(0.5 * log_var))
        x_hat, x_hat_var = self.decoder(z)
        return x_hat, x_hat_var, mean, log_var

    def sample(self, num_samples, device, mean = 0., sigma = 1.):
        latent_vector = mean + sigma * torch.randn(num_samples, self.latent_dims)
        samples, samples_var = self.decoder(latent_vector.to(device))

        return latent_vector, samples, samples_var

    def generate_states(self, x):
        return self.forward(x)[0]


