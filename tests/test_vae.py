import random
import torch
import numpy as np
from nosalro.vae import VariationalAutoencoder, StatesDataset, train, visualize, Scaler
import matplotlib.pyplot as plt

def random_points_generator(row, columns, dist, **kwargs):
    points = []
    for i in range(row):
        tmp_points = []
        for j in range(columns):
            tmp_points.append(dist(**kwargs))
        points.append(tmp_points)
    return np.array(points, dtype=np.float32)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.random.uniform(low=0, high=1, size = (1000,9)).astype(np.float32)
    # data = random_points_generator(10, 4, random.normalvariate, mu=0, sigma=1.)
    # dataset = StatesDataset(path='data/no_wall.dat', angle_to_sin_cos=True, angle_column=2)
    dataset = StatesDataset(x=data, angle_to_sin_cos=True, angle_column=2)

    # scaler = Scaler('standard')
    # scaler.fit(dataset.get_data())
    # dataset.scale_data(scaler)
    print(dataset.get_data().shape)

    vae = VariationalAutoencoder(10, 10, hidden_sizes=[64,64]).to(device)
    epochs = 2000
    lr = 3e-4
    vae = train(
        vae,
        epochs,
        lr,
        dataset,
        device,
        beta = 1,
        file_name = '.tmp/test.pt',
        overwrite = True,
        weight_decay = 0,
        batch_size = 128
    )
    i, x_hat_var, mu, logvar = vae(torch.tensor(dataset.get_data()).to(device), device, True, False)

    plt.errorbar(np.arange(10), i.mean(dim=0).detach().numpy(), x_hat_var.exp().mean(dim=0).detach().numpy(), linestyle='None', marker='^')
    plt.show(block=False)

    visualize([dataset.get_data(), i.detach().cpu().numpy()], projection='2d')
    print(i.mean(dim=0))
    print(x_hat_var.exp().mean(dim=0))


    # visualize([mu.detach().cpu().numpy()], projection='2d')
