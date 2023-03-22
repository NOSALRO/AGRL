import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
from stable_baselines3.sac.policies import SACPolicy
from nosalro.env import KheperaWithControllerEnv, KheperaDVController
from nosalro.vae import VariationalAutoencoder, StatesDataset, train, visualize, Scaler, Shuffle, AngleToSinCos, Transform
from nosalro.rl import learn
import pyfastsim as fastsim

if __name__ == '__main__':
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    shuffle = Shuffle(seed=42)
    angle_to_sin_cos = AngleToSinCos(angle_column=2)
    scaler = Scaler()
    target_scaler = Scaler()

    transforms = Transform([
        # shuffle,
        angle_to_sin_cos,
        scaler.fit,
        scaler
        ])

    target_transforms = Transform([
        angle_to_sin_cos,
        # target_scaler.fit,
        # target_scaler
        ])

    dataset = StatesDataset(path='data/no_wall.dat')
    target_dataset = StatesDataset(path='data/no_wall.dat')

    dataset.set_data(transforms(dataset[:]))
    target_dataset.set_data(target_transforms(target_dataset[:]))

    vae = VariationalAutoencoder(4, 2, output_dims=3, hidden_sizes=[32,32], scaler=scaler).to(device)
    epochs = 500
    lr = 3e-04
    vae = train(
        vae,
        epochs,
        lr,
        dataset,
        device,
        beta = 10,
        file_name = 'models/vae_models/no_wall_vae.pt',
        overwrite = False,
        weight_decay = 0,
        batch_size = 1024,
        target_dataset=target_dataset
    )

    # i, x_hat_var, mu, logvar = vae(torch.tensor(dataset[:]).to(device), device, False, scale=False)
    # visualize([dataset[:], i.detach().cpu().numpy()], projection='2d')

    # Env Init.
    world_map = fastsim.Map('worlds/no_wall.pbm', 600)
    robot = fastsim.Robot(10, fastsim.Posture(100., 100., 0.))

    action_space = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)

    observation_space = gym.spaces.Box(
    low=np.array([0, 0, -1, -1]),
    high=np.array([600, 600, 1, 1]),
    shape=(4,),
    dtype=np.float32
    )

    start_space = gym.spaces.Box(
    low=np.array([25, 25, -np.pi]),
    high=np.array([575, 575, np.pi]),
    shape=(3,),
    dtype=np.float32
    )

    env = KheperaDVController(
    robot=robot,
    world_map=world_map,
    reward_type='mse',
    target=False,
    n_obs=4,
    goals=target_dataset[[100]],
    goal_conditioned_policy=False,
    latent_rep=False,
    observation_space=observation_space,
    action_space=action_space,
    random_start=start_space,
    max_steps=0,
    vae=vae,
    scaler=scaler,
    )

    # # Agent Train.
    learn(env, device)