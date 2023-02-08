import torch

class VariationalEncoder(torch.nn.Module):

    def __init__(self, input_dims, latent_dims, hidden_sizes=[256, 128]):

        super().__init__()
        self.l1 = torch.nn.Linear(input_dims, hidden_sizes[0])
        self.l2 = torch.nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.l3 = torch.nn.Linear(hidden_sizes[1], latent_dims)
        self.l4 = torch.nn.Linear(hidden_sizes[1], latent_dims)
        self.drp1 = torch.nn.Dropout(0.2)

    def forward(self, x):

        x = torch.relu(self.l1(x))
        # x = self.drp1(x)
        x = torch.relu(self.l2(x))
        mu = self.l3(x)
        log_var = self.l4(x)

        return mu, log_var

class Decoder(torch.nn.Module):

    def __init__(self, input_dims, latent_dims, hidden_sizes=[256, 128]):

        super().__init__()
        self.l1 = torch.nn.Linear(latent_dims, hidden_sizes[-1])
        self.l2 = torch.nn.Linear(hidden_sizes[-1], hidden_sizes[-2])
        self.l3 = torch.nn.Linear(hidden_sizes[-2], input_dims)
        self.l4 = torch.nn.Linear(hidden_sizes[-2], input_dims)

        self.drp1 = torch.nn.Dropout(0.3)

    def forward(self, z):

        z = torch.relu(self.l1(z))
        # z = self.drp1(z) 
        z = torch.relu(self.l2(z))
        mu = self.l3(z)
        log_var = self.l4(z)

        return mu, log_var

class VAE(torch.nn.Module):

    def __init__(self, input_dims, latent_dims, hidden_sizes=[512,256]):

        super().__init__()
        self.encoder = VariationalEncoder(input_dims, latent_dims, hidden_sizes)
        self.decoder = Decoder(input_dims, latent_dims, hidden_sizes)

    def forward(self, x, device, deterministic=False):

        mu, log_var = self.encoder(x)
        z = mu if deterministic else self.reparameterizate(mu, log_var, device)
        x_hat, x_hat_var = self.decoder(z)
        return x_hat, x_hat_var, mu, log_var


    def reparameterizate(self, mu, log_var, device):

        epsilon = torch.autograd.Variable(torch.FloatTensor(log_var.size()).normal_()).to(device)
        std = log_var.mul(0.5).exp_()
        z = epsilon.mul(std).add_(mu)
        return z