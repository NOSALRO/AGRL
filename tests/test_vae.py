import random
import torch
import numpy as np
from nosalro.vae import VariationalAutoencoder, StatesDataset, train, visualize, Scaler

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
    # data = np.random.uniform(low=-1, high=1, size = (1000,4)).astype(np.float32)
    data = random_points_generator(100000, 4, random.normalvariate, mu=0, sigma=1.)
    # dataset = StatesDataset(path='data/no_wall.dat', angle_to_sin_cos=True, angle_column=2)
    dataset = StatesDataset(x=data, angle_to_sin_cos=False, angle_column=2)
    # scaler = Scaler('standard')
    # scaler.fit(dataset.get_data())
    # dataset.scale_data(scaler)
    vae = VariationalAutoencoder(4, 2, hidden_sizes=[64,64]).to(device)
    epochs = 100
    lr = 1e-4
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
    i, x_hat_var, mu, logvar = vae(torch.tensor(dataset.get_data()).to(device), device, False, False)
    visualize([dataset.get_data(), i.detach().cpu().numpy(), x_hat_var.detach().cpu().numpy()], projection='2d')
