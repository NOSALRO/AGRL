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
    data = np.random.uniform(low=50, high=550, size=(10000,4))
    # dataset = StatesDataset(path='data/no_wall.dat', angle_to_sin_cos=True, angle_column=2)
    dataset = StatesDataset(x=data, angle_to_sin_cos=False, angle_column=2)

    scaler = Scaler('standard')
    scaler.fit(dataset.get_data())
    dataset.scale_data(scaler)

    vae = VariationalAutoencoder(4, 4, hidden_sizes=[32,32]).to(device)
    epochs = 300
    lr = 1e-03
    vae = train(
        vae,
        epochs,
        lr,
        dataset,
        device,
        beta = 0.3,
        file_name = '.tmp/test.pt',
        overwrite = True,
        weight_decay = 0,
        batch_size = 512
    )
    i, x_hat_var, mu, logvar = vae(torch.tensor(dataset.get_data()).to(device), device, True, False)

    visualize([dataset.get_data(), i.detach().cpu().numpy()], projection='2d')
    print(i.mean(dim=0))
    print(x_hat_var.exp().mean(dim=0))


    # visualize([mu.detach().cpu().numpy()], projection='2d')
