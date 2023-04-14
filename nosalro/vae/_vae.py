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
        log_var = self.layers[-1](x)
        return mu, log_var

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
        log_var = self.layers[-1](x)
        return mu, log_var

class VariationalAutoencoder(torch.nn.Module):

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
        self.encoder = VariationalEncoder(self.input_dims, self.latent_dims, copy.copy(hidden_sizes))
        self.decoder = VariationalDecoder(self.latent_dims, self.output_dims, copy.copy(hidden_sizes))

    def forward(self, x, device, deterministic=False, scale=False):
        x = torch.tensor(self.scaler(x.cpu()), device=device) if scale else x
        mu, log_var = self.encoder(x)
        z = mu if deterministic else self.reparameterizate(mu, log_var, device)
        x_hat, x_hat_var = self.decoder(z)
        return x_hat, x_hat_var, mu, log_var

    def reparameterizate(self, mu, log_var, device):
        epsilon = torch.autograd.Variable(torch.FloatTensor(log_var.size()).normal_()).to(device)
        std = log_var.mul(0.5).exp_()
        z = epsilon.mul(std).add_(mu)
        return z

    def sample(self, num_samples, device, mean = 0., sigma = 1.):
        latent_vector = mean + sigma * torch.randn(num_samples, self.latent_dims)
        if num_samples == 1:
            latent_vector = latent_vector.squeeze()
        samples, samples_var = self.decoder(latent_vector.to(device))
        return latent_vector, samples, samples_var
