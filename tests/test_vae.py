import torch
from nosalro.vae import VariationalAutoencoder, StatesDataset, train, visualize

if __name__ == '__main__':
    dataset = StatesDataset(path="data/no_wall.dat", angle_to_sin_cos=False, angle_column=2)
    _min, _max = dataset.scale_data()

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=True
    )

    vae = VariationalAutoencoder(3, 2, min=_min, max=_max)
    epochs = 1
    lr = 5e-4
    vae = train(
        vae,
        epochs,
        lr,
        dataloader,
        'cpu',
        beta = 0.1,
        file_name='model/model/vae.pt',
        overwrite=True,
        beta=1,
        weight_decay=0
    )
    visualize(dataset.get_data(), projection='3d')
