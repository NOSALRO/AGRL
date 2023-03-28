import pickle
import numpy as np
import torch
import gym
from nosalro.env import KheperaEnv, Box
from nosalro.vae import StatesDataset, VariationalAutoencoder, train, visualize
from nosalro.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import pyfastsim as fastsim


with open(".tmp/test/env.pkl",'rb') as file:
    env = pickle.load(file)

env.reset()
env.render()
while True:
    _, _, done, _ = env.step([1,1])
    if done:
        env.reset()
