import numpy as np
import torch
from agrl.env import SimpleEnv
from agrl.robots import Point
from agrl.vae import StatesDataset, VariationalAutoencoder, Scaler, train, visualize

robot = Point([10, 10])
dataset = StatesDataset(path="data/pdp_data.dat", angle_to_sin_cos=False, angle_column=2)
scaler = Scaler('standard')
scaler.fit(dataset.get_data())
dataset.scale_data(scaler)
print(dataset.get_data())