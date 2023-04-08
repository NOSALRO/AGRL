import numpy as np
import torch
import gym
import matplotlib.pyplot as plt
from stable_baselines3.sac.policies import SACPolicy
from nosalro.env import KheperaWithControllerEnv
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
        shuffle,
        angle_to_sin_cos,
        scaler.fit,
        scaler
        ])

    target_transforms = Transform([
        shuffle,
        angle_to_sin_cos,
        target_scaler.fit,
        target_scaler
        ])

    dataset = StatesDataset(path='data/no_wall.dat')
    target_dataset = StatesDataset(path='data/no_wall.dat')

    dataset.set_data(transforms(dataset[:]))
    target_dataset.set_data(target_transforms(target_dataset[:]))

    vae = VariationalAutoencoder(4, 2, output_dims=4, hidden_sizes=[32,32], scaler=scaler).to(device)
    epochs = 500
    lr = 3e-04
    vae = train(
        vae,
        epochs,
        lr,
        dataset,
        device,
        beta = 10,
        file_name = 'models/vae_models/no_wall_vae_cos_sin.pt',
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
    target_dataset.set_data(target_scaler(target_dataset[:], undo=True))

    action_space = gym.spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)

    observation_space = gym.spaces.Box(
    low=np.array([0, 0, -1, -1, -np.inf, -np.inf]),
    high=np.array([600, 600, 1, 1, np.inf, np.inf]),
    shape=(6,),
    dtype=np.float32
    )

    start_space = gym.spaces.Box(
    low=np.array([25, 25, -np.pi]),
    high=np.array([575, 575, np.pi]),
    shape=(3,),
    dtype=np.float32
    )

    env = KheperaWithControllerEnv(
    robot=robot,
    world_map=world_map,
    reward_type='edl',
    target=False,
    n_obs=4,
    goals=target_dataset[:],
    goal_conditioned_policy=True,
    latent_rep=True,
    observation_space=observation_space,
    action_space=action_space,
    random_start=start_space,
    max_steps=0,
    vae=vae,
    scaler=scaler,
    )

    # # Agent Train.
    learn(env, device)
