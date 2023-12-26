import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import copy

def loss_fn(x_target, x_hat, x_hat_var, mu, log_var, beta, reconstruction_type='mse', lim_logvar=False):
    if lim_logvar:
        max_logvar = 2
        min_logvar = -2
        x_hat_var = max_logvar - torch.nn.functional.softplus(max_logvar - x_hat_var)
        x_hat_var = min_logvar + torch.nn.functional.softplus(x_hat_var - min_logvar)
    gnll = torch.nn.functional.gaussian_nll_loss(x_hat, x_target, x_hat_var.exp(), reduction='sum', full=False)
    mse = torch.nn.functional.mse_loss(x_hat, x_target, reduction='sum')
    # p = torch.distributions.Normal(torch.zeros_like(mu).to('cuda'), torch.ones_like(log_var).to('cuda'))
    # q = torch.distributions.Normal(mu, log_var.mul(0.5).exp())
    # kld = beta * torch.distributions.kl_divergence(p,q).sum()
    kld = beta * torch.mean(-0.5 * torch.sum(1. + log_var - mu.pow(2) - log_var.exp(), dim=1))
    if reconstruction_type == 'mse':
        return mse+kld, mse, kld
    elif reconstruction_type == 'gnll':
        return gnll+kld, gnll, kld

def train(
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
        reconstruction_type='mse',
        lim_logvar=False
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
                    x_hat, x_hat_var, mu, log_var = model(x, device)
                    loss, _reconstruction_loss, _kld = loss_fn(target, x_hat, x_hat_var, mu, log_var, beta, reconstruction_type, lim_logvar)
                elif len(data) == 1:
                    x = data[0]
                    x = x.to(device)
                    x_hat, x_hat_var, mu, log_var = model(x, device)
                    loss, _reconstruction_loss, _kld = loss_fn(x, x_hat, x_hat_var, mu, log_var, beta, reconstruction_type, lim_logvar)
                loss.backward()
                losses.append([loss.cpu().detach(), _reconstruction_loss.cpu().detach(), _kld.cpu().detach()])
                optimizer.step()
            epoch_loss = np.mean(losses, axis=0)
            print(f"Epoch: {epoch}: ",
                f"| Reconstruction -> {epoch_loss[1]:.8f} ",
                f"| KL Div -> {epoch_loss[2]:.8f} ",
                f"| Total loss -> {epoch_loss[0]:.8f}",)
        os.makedirs('/'.join(file_name.split('/')[:-1]), exist_ok=True)
        torch.save(model, file_name)
    return model

def visualize(data, projection='2d', color=None, columns_select=None, *, file_name=None):
    fig = plt.figure()
    projection = projection if projection != '2d' and not None else None
    ax = plt.axes(projection=projection)
    for idx, x in enumerate(data):
        ax.set_xlim(0,600)
        ax.set_ylim(0,600)
        columns = x.shape[-1] if columns_select is None else len(columns_select)
        if x.shape[-1] >= 3:
            c = color
            if color is None:
                c = x[:,2]
            if projection == '3d':
                state_space = ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], c=c, cmap='coolwarm')
            else:
                state_space = ax.scatter(x[:, 0], x[:, 1], c=c, cmap='coolwarm')
            fig.colorbar(state_space, shrink=0.8, aspect=5)
        elif x.shape[-1] == 2:
            state_space = ax.scatter(x[:, 0], x[:, 1], cmap='coolwarm')
    if file_name is not None:
        plt.savefig(f'{file_name}.png')
    if file_name is None:
        plt.show(block=True)