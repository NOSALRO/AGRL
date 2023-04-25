import copy
import torch
from torch.autograd import Variable


class CategoricalEncoder(torch.nn.Module):

    def __init__(
            self,
            input_dim,
            latent_dim,
            hidden_sizes=[256, 128]
        ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.layers = torch.nn.ModuleList()
        layers_size = hidden_sizes
        layers_size.insert(0, self.input_dim)
        layers_size.insert(len(layers_size), self.latent_dim)

        for size_idx in range(1, len(layers_size)):
            self.layers.append(
                torch.nn.Linear(layers_size[size_idx-1] , layers_size[size_idx])
                )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        out = self.layers[-1](x)
        return out

class CategoricalDecoder(torch.nn.Module):

    def __init__(
            self,
            latent_dim,
            output_dim,
            hidden_sizes=[256, 128]
        ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.layers = torch.nn.ModuleList()
        layers_size = hidden_sizes
        layers_size.insert(0, self.output_dim)
        layers_size.insert(len(layers_size), self.latent_dim)
        layers_size.reverse()

        for size_idx in range(1, len(layers_size)):
            self.layers.append(
                torch.nn.Linear(layers_size[size_idx-1] , layers_size[size_idx])
                )
            if size_idx == len(layers_size) - 1:
                self.layers.append(
                    torch.nn.Linear(layers_size[size_idx-1] , layers_size[size_idx])
                    )

    def forward(self, x):
        for layer in self.layers[:-2]:
            x = torch.tanh(layer(x))
        mu = self.layers[-2](x)
        log_var = self.layers[-1](x)
        return mu, log_var

class CategoricalVAE(torch.nn.Module):

    def __init__(
            self,
            input_dims,
            latent_dims,
            hidden_sizes = [256, 128],
            scaler = None,
            output_dims = None,
        ):
        super().__init__()
        self.scaler = scaler
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.output_dims = output_dims if output_dims is not None else input_dims
        self.encoder = CategoricalEncoder(self.input_dims, self.latent_dims, copy.copy(hidden_sizes))
        self.decoder = CategoricalDecoder(self.latent_dims, self.output_dims, copy.copy(hidden_sizes))

    def forward(self, x, device, deterministic=False, scale=False):
        x = torch.tensor(self.scaler(x.cpu()), device=device) if scale else x
        latent = self.encoder(x)
        z = torch.softmax(latent, dim=-1) if deterministic else self.reparameterizate(latent, device)
        x_hat, x_hat_var = self.decoder(z)
        return x_hat, x_hat_var, latent

    def reparameterizate(self, latent, device):
        eps = 1e-20
        uni_sample = torch.rand(size=latent.size())
        gumbel_sample = -torch.log(-torch.log(uni_sample + eps) + eps)
        gumbel_sample = gumbel_sample.to(device)
        z = latent + gumbel_sample
        z = torch.nn.functional.softmax(z, dim=-1)
        return z

    def sample(self, num_samples, device, mean = 0., sigma = 1.):
        latent_vector = mean + sigma * torch.randn(num_samples, self.latent_dims)
        samples, samples_var = self.decoder(latent_vector.to(device))
        return latent_vector, samples, samples_var