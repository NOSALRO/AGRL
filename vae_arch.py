import torch
import torch.nn.functional as F
import numpy as np

class VarEncoder(torch.nn.Module):

    def __init__(self, input_shape, latent_dims) -> None:

        super().__init__()

        self.l1 = torch.nn.Linear(input_shape, 9)
        self.l2 = torch.nn.Linear(9, 5)
        self.l3 = torch.nn.Linear(5, latent_dims)
        self.l4 = torch.nn.Linear(5, latent_dims)

    
    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        mu = self.l3(x)
        log_var = self.l4(x)

        return mu, log_var

class Decoder(torch.nn.Module):

    def __init__(self, input_shape, latent_dims) -> None:
        super().__init__()

        self.l1 = torch.nn.Linear(latent_dims, 5)
        self.l2 = torch.nn.Linear(5, 9)
        self.l3 = torch.nn.Linear(9, input_shape)

    def forward(self, x):

        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        return self.l3(x)

class VarAutoencoder(torch.nn.Module):

    def __init__(self, input_shape, latent_dims) -> None:
        super().__init__()

        self.latent_dims = latent_dims
        self.encoder = VarEncoder(input_shape, latent_dims)
        self.decoder = Decoder(input_shape, latent_dims)

    # Reparameterization trick to backprop stochastic node. σ * ε + μ
    def reparam(self, mean, var):
        eps = torch.randn_like(var)
        z = mean + eps*var
        return z 
    
    def forward(self, x):

        mean, log_var = self.encoder(x)
        z = self.reparam(mean, torch.sqrt(torch.exp(log_var)))
        x_hat = self.decoder(z)
        return x_hat, mean, log_var
    
    def sample(self, num_samples, device):

        latent_vector = torch.randn(num_samples, self.latent_dims)
        samples = self.decoder(latent_vector.to(device))

        return latent_vector, samples
    
    def generate_states(self, x):

        return self.forward(x)[0]

        
