import torch
import torch.nn.functional as F

class VarEncoder(torch.nn.Module):

    def __init__(self, input_shape, latent_dims) -> None:

        super().__init__()

        self.l1 = torch.nn.Linear(input_shape, 9)
        self.l2 = torch.nn.Linear(9, 5)
        self.l3 = torch.nn.Linear(5, latent_dims)
        self.l4 = torch.nn.Linear(5, latent_dims)
        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    
    def forward(self, x):
        x = F.leaky_relu(self.l1(x))
        x = F.leaky_relu(self.l2(x))
        mu = self.l3(x)
        sigma = torch.exp(self.l4(x))

        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = .5*(sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    

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

        self.encoder = VarEncoder(input_shape, latent_dims)
        self.decoder = Decoder(input_shape, latent_dims)
    
    def forward(self, x):

        z = self.encoder(x)
        return self.decoder(z)