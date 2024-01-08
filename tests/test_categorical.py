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
from agrl.vae import CategoricalVAE, StatesDataset, train, visualize
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shuffle = Shuffle(seed=42)
    scaler = Scaler()
    transforms = Compose([shuffle, scaler.fit, scaler])

    # dataset = StatesDataset(path='data/alley_right_go_explore.dat', transforms=transforms)
    x = np.linspace(0, 600, 50)
    y = np.linspace(0, 600, 50)
    grid = []
    for i in x:
        for j in y:
            grid.append([i,j])
    grid = np.array(grid)
    dataset = StatesDataset(grid, transforms=transforms)

    cvae = CategoricalVAE(2, 2, output_dims=2, hidden_sizes=[128,128], scaler=scaler).to(device)
    epochs = 200
    lr = 3e-04
    vae = cvae.train(
        cvae,
        epochs,
        lr,
        dataset,
        device,
        beta = 1,
        file_name = '.tmp/categorical_vae.pt',
        overwrite = True,
        weight_decay = 0,
        batch_size = 256,
    )

    x_hat, x_hat_var, l = vae(torch.tensor(dataset[:]).to(device), device, True, False)
    visualize([scaler(dataset[:], undo=True), scaler(x_hat.detach().cpu().numpy(), undo=True)], projection='2d')
    # print(torch.softmax(l[:5], dim=-1))

    one_hot = np.eye(10).astype(np.float32)
    # print(one_hot.shape)

    x_hat, x_hat_var = vae.decoder(torch.tensor(one_hot).to(device))
    visualize([scaler(x_hat.detach().cpu().numpy(), undo=True)], projection='2d')