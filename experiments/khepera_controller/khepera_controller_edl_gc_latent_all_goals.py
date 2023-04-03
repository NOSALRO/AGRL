import numpy as np
import torch
import gym
from nosalro.env import KheperaControllerEnv, Box
from nosalro.vae import StatesDataset, VariationalAutoencoder, train, visualize
from nosalro.transforms import Compose, AngleToSinCos, Scaler, Shuffle
from nosalro.rl.utils import train_sb3, cli
from nosalro.controllers import Controller
import pyfastsim as fastsim


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = Scaler()
    target_scaler = Scaler(_type='standard')
    shuffle = Shuffle(seed=42)

    transforms = Compose([
        shuffle,
        AngleToSinCos(angle_column=2),
        scaler.fit,
        scaler
    ])
    dataset = StatesDataset(path='data/go_explore_1000_fix.dat', transforms=transforms)

    # In case output dims is different from input dims.
    target_transforms = Compose([
        shuffle,
        target_scaler.fit,
        target_scaler
    ])
    target_dataset = StatesDataset(path='data/go_explore_1000_fix.dat', transforms=target_transforms)

    vae = VariationalAutoencoder(input_dims=4, latent_dims=3, output_dims=3, hidden_sizes=[64,64], scaler=scaler).to(device)
    vae = train(
        model = vae,
        epochs = 3000,
        lr = 1e-04,
        dataset = dataset,
        device = device,
        beta = 10,
        file_name = 'models/vae_models/dots_vae_cos_sin.pt',
        overwrite = False,
        weight_decay = 0,
        batch_size = 1024,
        target_dataset = target_dataset
    )
    # i, x_hat_var, mu, logvar = vae(torch.tensor(dataset[:]).to(device), device, True, scale=False)
    # visualize([dataset[:], i.detach().cpu().numpy()], projection='2d')

    # Env Init.
    world_map = fastsim.Map('worlds/dots.pbm', 600)
    robot = fastsim.Robot(10, fastsim.Posture(500., 500., 0.))
    dataset.inverse([scaler])
    target_dataset.inverse([target_scaler])

    action_space = Box(low=-1., high=1., shape=(2,), dtype=np.float32)
    observation_space = Box(
        low=np.array([0, 0,  -1, -1, -np.inf, -np.inf, -np.inf]),
        high=np.array([600, 600, 1, 1, np.inf, np.inf, np.inf]),
        shape=(7,),
        dtype=np.float32
    )
    start_space = Box(
        low=np.array([25, 25, -np.pi]),
        high=np.array([575, 575, np.pi]),
        shape=(3,),
        dtype=np.float32
    )

    env = KheperaControllerEnv(
        robot=robot,
        world_map=world_map,
        reward_type='edl',
        n_obs=4,
        goals=dataset[:],
        goal_conditioned_policy=True,
        latent_rep=True,
        observation_space=observation_space,
        action_space=action_space,
        random_start=False,
        max_steps=0,
        vae=vae,
        scaler=target_scaler,
        controller=Controller(0.5, 0)
    )

    # # Agent Train.
    train_sb3(env, device, **cli())