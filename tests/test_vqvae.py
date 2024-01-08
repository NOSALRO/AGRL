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
import os
import copy
from agrl.vae import StatesDataset
from agrl.transforms import Compose, AngleToSinCos, Scaler, Shuffle
from agrl.vae import VQVAE
from agrl.vae import visualize
import numpy as np
import torch
import matplotlib.pyplot as plt



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = Scaler()
    shuffler = Shuffle()
    transforms = Compose([shuffler, scaler.fit, scaler])
    dataset = StatesDataset(path='data/go_explore_xy.dat', transforms=transforms)

    vqvae = VQVAE(input_dims=2, latent_dims=8, codes=10, output_dims=2, beta=0.25, hidden_sizes = [64, 64])
    vqvae = vqvae.to(device)

    vqvae = vqvae.train(
        loss_fn=vqvae.loss_fn,
        model=vqvae,
        epochs=700,
        lr=3e-4,
        dataset = dataset,
        device = device,
        overwrite=False,
        file_name = '.tmp/vqvae_test',
        batch_size = 256,
        reconstruction_type='mse'
    )

    x_hat, x_hat_var, one_hot, _ = vqvae(torch.tensor(dataset[:], device=device), device, False, False)
    print(one_hot)
    x_hat = x_hat.detach().cpu().numpy()
    visualize([scaler(dataset[:], True), scaler(x_hat[:], True)], file_name='.tmp/plot')
