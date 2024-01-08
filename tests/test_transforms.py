#!/usr/bin/env python
# encoding: utf-8
#|
#|    Copyright (c) 2022-2023 Computational Intelligence Lab, University of Patras, Greece
#|    Copyright (c) 2022-2023 Constantinos Tsakonas
#|    Copyright (c) 2022-2023 Konstantinos Chatzilygeroudis
#|    Authors:  Constantinos Tsakonas
#|              Konstantinos Chatzilygeroudis
#|    email:    tsakonas.constantinos@gmail.com
#|              costashatz@gmail.com
#|    website:  https://nosalro.github.io/
#|              http://cilab.math.upatras.gr/
#|
#|    This file is part of AGRL.
#|
#|    All rights reserved.
#|
#|    Redistribution and use in source and binary forms, with or without
#|    modification, are permitted provided that the following conditions are met:
#|
#|    1. Redistributions of source code must retain the above copyright notice, this
#|       list of conditions and the following disclaimer.
#|
#|    2. Redistributions in binary form must reproduce the above copyright notice,
#|       this list of conditions and the following disclaimer in the documentation
#|       and/or other materials provided with the distribution.
#|
#|    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#|    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#|    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#|    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#|    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#|    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#|    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#|    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#|    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#|    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#|
import sys
import os
import random
import torch
import numpy as np
from agrl.vae import VariationalAutoencoder, StatesDataset, train, visualize
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

    shuffle = Shuffle(seed=42)
    angle_to_sin_cos = AngleToSinCos(angle_column=2)
    scaler = Scaler()
    target_scaler = Scaler()

    transforms = Compose([
        shuffle,
        angle_to_sin_cos,
        scaler.fit,
        scaler
        ])

    target_transforms = Compose([
        shuffle,
        # angle_to_sin_cos,
        target_scaler.fit,
        target_scaler
        ])

    dataset = StatesDataset(path='data/no_wall.dat', transforms=transforms)
    target_dataset = StatesDataset(path='data/no_wall.dat', transforms=target_transforms)

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
        overwrite = True,
        weight_decay = 0,
        batch_size = 1024,
        target_dataset=target_dataset
    )

    i, x_hat_var, mu, logvar = vae(torch.tensor(dataset[:]).to(device), device, False, scale=False)
    visualize([dataset[:], i.detach().cpu().numpy()], projection='2d')
