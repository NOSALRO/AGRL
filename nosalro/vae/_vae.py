import copy
import torch


class VariationalEncoder(torch.nn.Module):

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
            if size_idx == len(layers_size) - 1:
                self.layers.append(
                    torch.nn.Linear(layers_size[size_idx-1] , layers_size[size_idx])
                    )

    def forward(self, x):
        for layer in self.layers[:-2]:
            x = torch.relu(layer(x))
        mu = self.layers[-2](x)
        log_std = self.layers[-1](x)
        return mu, log_std

class VariationalDecoder(torch.nn.Module):

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
            x = torch.relu(layer(x))
        mu = self.layers[-2](x)
        log_std = self.layers[-1](x)
        return mu, log_std

class VariationalAutoencoder(torch.nn.Module):

    def __init__(
            self,
            input_dims,
            latent_dims,
            hidden_sizes = [512,256],
            scaler = None,
        ):
        super().__init__()
        self.scaler = scaler
        self.encoder = VariationalEncoder(input_dims, latent_dims, copy.copy(hidden_sizes))
        self.decoder = VariationalDecoder(latent_dims, input_dims, copy.copy(hidden_sizes))

    def forward(self, x, device, deterministic=False, scale=False):
        x = self.scaler(x) if scale else x
        mu, log_std = self.encoder(x)
        z = mu if deterministic else self.reparameterizate(mu, log_std, device)
        x_hat, x_hat_std = self.decoder(z)
        return x_hat, x_hat_std, mu, log_std

    def reparameterizate(self, mu, log_std, device):
        epsilon = torch.autograd.Variable(torch.FloatTensor(log_std.size()).normal_()).to(device)
        std = log_std.exp_()
        z = epsilon.mul(std).add_(mu)
        return z