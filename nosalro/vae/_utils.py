import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def loss_fn(x_target, x_hat, x_hat_var, mu, log_var, beta):
    gnll = torch.nn.functional.gaussian_nll_loss(x_hat, x_target, torch.exp(0.5 * x_hat_var))
    kld = beta * torch.mean(-0.5 * torch.sum(1. + log_var - mu**2 - torch.exp(.5 * log_var), dim=1))
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
        batch_size = 256
    ):
    if len(file_name.split('/')) == 1:
        file_name = 'models/' + file_name
    if len(file_name.split('/')[-1].split('.')) == 1:
        file_name = file_name + '.pt'
    try:
        if overwrite: raise FileNotFoundError
        model = torch.load(file_name)
        print("Loaded saved model.")
    except FileNotFoundError:
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(epochs):
            losses = []
            for x in dataloader:
                optimizer.zero_grad()
                x = x.to(device)
                x_hat, x_hat_var, mean, log_var = model(x, device)
                loss, _gnll, _kld = loss_fn(x, x_hat, x_hat_var, mean, log_var, beta)
                loss.backward()
                losses.append([loss.cpu().detach(), _gnll.cpu().detach(), _kld.cpu().detach()])
                optimizer.step()
            epoch_loss = np.mean(losses, axis=0)
            print(f"Epoch: {epoch}: ",
                f"| Gaussian NLL -> {epoch_loss[1]:.3f} ",
                f"| KL Div -> {epoch_loss[2]:.3f} ",
                f"| Total loss -> {epoch_loss[0]:.3f}",)
        os.makedirs('/'.join(file_name.split('/')[:-1]), exist_ok=True)
        torch.save(model, file_name)
    return model

def visualize(x, projection='2d', color=None, columns_select=None):
    fig = plt.figure()
    plt.ion()
    plt.show(block=False)
    projection = projection if projection != '2d' and not None else None
    ax = plt.axes(projection=projection)
    columns_select = x.shape[-1] if columns_select is None else len(columns_select)
    if x.shape[-1] >= 3:
        c = color
        if color is None:
            c = x[:,2]
        if projection == '3d':
            state_space = ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], c=c, cmap='coolwarm')
        else:
            state_space = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=c, cmap='coolwarm')
        fig.colorbar(state_space, shrink=0.8, aspect=5)
    elif x.shape[-1] == 2:
        state_space = ax.scatter(x[:, 0], x[:, 1], cmap='coolwarm')
    plt.draw()
    plt.pause(0.001)
    input('Press any key to continue')
    plt.close('all')