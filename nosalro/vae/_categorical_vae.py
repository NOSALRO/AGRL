import os
import copy
import torch
import numpy as np
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
        eps = 1e-10
        uni_sample = torch.rand(size=latent.size())
        gumbel_sample = -torch.log(-torch.log(uni_sample + eps) + eps)
        gumbel_sample = gumbel_sample.to(device)
        z = latent + gumbel_sample
        z = torch.nn.functional.softmax(z, dim=-1)
        return z/0.1

    def sample(self, num_samples, device, mean = 0., sigma = 1.):
        latent_vector = mean + sigma * torch.randn(num_samples, self.latent_dims)
        samples, samples_var = self.decoder(latent_vector.to(device))
        return latent_vector, samples, samples_var

    @staticmethod
    def loss_fn(x_target, x_hat, x_hat_var, latent, beta):
        eps = 1e-10
        gnll = torch.nn.functional.gaussian_nll_loss(x_hat, x_target, x_hat_var.exp(), reduction='sum')
        mse = torch.nn.functional.mse_loss(x_hat, x_target, reduction='mean')
        latent = torch.softmax(latent, dim=-1)
        q = latent * torch.log(latent + eps)
        p = latent*torch.log(torch.tensor(1/float(latent.size()[-1]))+ eps)
        kld = beta * torch.mean(torch.sum(q-p, dim=1), dim=0)
        return mse, gnll, kld

    def train(
            self,
            model,
            epochs,
            lr,
            dataset,
            device,
            file_name,
            overwrite = False,
            beta = 1,
            weight_decay = 0,
            batch_size = 256,
            target_dataset = None,
        ):
        if len(file_name.split('/')) == 1:
            file_name = 'models/' + file_name
        if len(file_name.split('/')[-1].split('.')) == 1:
            file_name = file_name + '.pt'
        try:
            if overwrite: raise FileNotFoundError
            model = torch.load(file_name, map_location=device)
            print("Loaded saved model.")
        except FileNotFoundError:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            dataloader = [torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)]
            if target_dataset is not None:
                target_dataloader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size=batch_size)
                dataloader.append(target_dataloader)

            for epoch in range(epochs):
                losses = []
                for data in zip(*dataloader):
                    optimizer.zero_grad()
                    if len(data) == 2:
                        x, target = data
                        x = x.to(device)
                        target = target.to(device)
                        x_hat, x_hat_var, latent = model(x, device)
                        loss, _gnll, _kld = closs_fn(target, x_hat, x_hat_var, latent, beta)
                    elif len(data) == 1:
                        x = data[0]
                        x = x.to(device)
                        x_hat, x_hat_var, latent = model(x, device)
                        loss, _gnll, _kld = self.loss_fn(x, x_hat, x_hat_var, latent, beta)
                    loss.backward()
                    losses.append([loss.cpu().detach(), _gnll.cpu().detach(), _kld.cpu().detach()])
                    optimizer.step()
                epoch_loss = np.mean(losses, axis=0)
                print(f"Epoch: {epoch}: ",
                    f"| Gaussian NLL -> {epoch_loss[1]:.8f} ",
                    f"| KL Div -> {epoch_loss[2]:.8f} ",
                    f"| Total loss -> {epoch_loss[0]:.8f}",)
            os.makedirs('/'.join(file_name.split('/')[:-1]), exist_ok=True)
            torch.save(model, file_name)
        return model
