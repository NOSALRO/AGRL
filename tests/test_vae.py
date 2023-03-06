import torch
from nosalro.vae import VariationalAutoencoder, StatesDataset, train, visualize, Scaler

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = StatesDataset(path="data/no_wall.dat", angle_to_sin_cos=False, angle_column=2)
    scaler = Scaler('standard')
    scaler.fit(dataset.get_data())
    dataset.scale_data(scaler)
    vae = VariationalAutoencoder(3, 2, scaler=scaler).to(device)
    epochs = 50
    lr = 1e-3
    vae = train(
        vae,
        epochs,
        lr,
        dataset,
        device,
        beta = 0,
        file_name = 'vae.pt',
        overwrite = False,
        weight_decay = 0,
        batch_size = 512
    )
    visualize(dataset.get_data(), projection='2d')
    visualize(vae(torch.tensor(dataset.get_data()).to(device), device, True, False)[0].detach().cpu().numpy(), projection='2d')
