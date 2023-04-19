import torch
import numpy as np
from nosalro.env import KheperaDVControllerEnv, Box
from nosalro.controllers import DVController
from nosalro.rl.td3 import Actor, Critic, train_td3
from nosalro.vae import StatesDataset, VariationalAutoencoder, train, visualize
from nosalro.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import pyfastsim as fastsim


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

scaler = Scaler()
shuffle = Shuffle(seed=42)
transforms = Compose([shuffle, scaler.fit, scaler])

# dataset = StatesDataset(np.loadtxt('data/alley_right_go_explore.dat')[:,:2], transforms=transforms)
dataset = StatesDataset(path='data/uniform_alley.dat', transforms=transforms)

vae = VariationalAutoencoder(input_dims=2, latent_dims=1, output_dims=2, hidden_sizes=[64,64], scaler=scaler).to(device)
vae = train(
    model = vae,
    epochs = 1000,
    lr = 7e-05,
    dataset = dataset,
    device = device,
    beta = 20,
    # file_name = 'models/vae_models/alley_vae_1d_right.pt',
    file_name = 'models/vae_models/vae_uniform.pt',
    overwrite = False,
    weight_decay = 0,
    batch_size = 64,
    # target_dataset = target_dataset
)
goals = []
for i in dataset:
    for _ in range(5):
        i, x_hat_var, mu, log_var = vae(torch.tensor(i).to(device), device, True, scale=False)
        _, out, _ = vae.sample(1, device, mu.cpu(), log_var.mul(0.5).exp().cpu())
        goals.append(scaler(out.cpu().detach().squeeze().numpy(), undo=True))
# visualize([np.array(goals)], projection='2d', file_name='.tmp/image_ge')
# visualize([np.array(goals)], projection='2d', file_name='.tmp/image_uniform')
world_map = fastsim.Map('worlds/alley_right.pbm', 600)
disp = fastsim.Display(world_map, fastsim.Robot(10, fastsim.Posture(0, 0,0)))
for i in goals:
    world_map.add_goal(fastsim.Goal(*i[:2], 10, 1))
while True:
    disp.update()

# i, x_hat_var, mu, logvar = vae(torch.tensor(dataset[:]).to(device), device, True, scale=False)
# visualize([dataset[:], i.detach().cpu().numpy()], projection='2d', file_name='.tmp/image')
