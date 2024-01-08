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
import torch
import numpy as np

class Encoder(torch.nn.Module):

    def __init__(
            self,
            input_dim,
            latent_dim,
            hidden_sizes=[256, 128]
        ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.layers = torch.nn.ModuleList()
        layers_size = hidden_sizes
        layers_size.insert(0, self.input_dim)
        layers_size.insert(len(layers_size), self.latent_dim)

        for size_idx in range(1, len(layers_size)):
            self.layers.append(
                torch.nn.Linear(layers_size[size_idx-1] , layers_size[size_idx])
                )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        latent = self.layers[-1](x)
        return latent

class VectorQuantizer(torch.nn.Module):
    def __init__(self,
                 K: int,
                 D: int,
                 beta: float = 0.25):
        super().__init__()
        self.K = K
        self.D = D
        self.beta = beta

        self.embedding = torch.nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents):
        latents = latents.contiguous()
        latents_shape = latents.shape
        # Compute L2 distance between latents and embedding weights
        center_distance = torch.sum(latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(latents, self.embedding.weight.T)

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(center_distance, dim=1).unsqueeze(1)

        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=latents.device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)

        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)
        quantized_latents = quantized_latents.view(latents_shape)
        # https://arxiv.org/pdf/1711.00937.pdf -> EQ(3)
        commitment_loss = torch.nn.functional.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = torch.nn.functional.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Backprop trick. latent - latents = 0 however gradients are copied.
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.contiguous(), vq_loss, encoding_one_hot


class Decoder(torch.nn.Module):

    def __init__(
            self,
            latent_dim,
            output_dim,
            hidden_sizes=[256, 128]
        ):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.layers = torch.nn.ModuleList()
        layers_size = hidden_sizes
        layers_size.insert(0, self.output_dim)
        layers_size.insert(len(layers_size), self.latent_dim)
        layers_size.reverse()

        for size_idx in range(1, len(layers_size)):
            self.layers.append(
                torch.nn.Linear(layers_size[size_idx-1] , layers_size[size_idx])
                )
            if size_idx == len(layers_size) - 1:
                self.layers.append(
                    torch.nn.Linear(layers_size[size_idx-1] , layers_size[size_idx])
                    )

    def forward(self, x):
        for layer in self.layers[:-2]:
            x = torch.relu(layer(x))
        mu = self.layers[-2](x)
        log_var = self.layers[-1](x)
        return mu, log_var

class VQVAE(torch.nn.Module):

    def __init__(
            self,
            input_dims,
            latent_dims,
            codes,
            hidden_sizes = [256, 128],
            scaler = None,
            output_dims = None,
            beta = 1.25
        ):
        super().__init__()
        self.scaler = scaler
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.codes = codes
        self.output_dims = output_dims if output_dims is not None else input_dims
        self.encoder = Encoder(self.input_dims, self.latent_dims, copy.copy(hidden_sizes))
        self.vq = VectorQuantizer(self.codes, self.latent_dims, beta)
        self.decoder = Decoder(self.latent_dims, self.output_dims, copy.copy(hidden_sizes))

    def forward(self, x, device, deterministic=False, scale=False):
        if self.scaler is not None:
            x = torch.tensor(self.scaler(x.cpu()), device=device) if scale else x
        x = x.view(1,-1) if len(x.size()) < 2 else x
        latents = self.encoder(x)
        z, vq_loss, one_hot = self.vq(latents)
        x_hat, x_hat_var = self.decoder(z)
        return x_hat, x_hat_var, one_hot, vq_loss

    @staticmethod
    def loss_fn(x_target, x_hat, x_hat_var, vq_loss, reconstruction_type='mse', lim_logvar=False):
        if lim_logvar:
            max_logvar = 2
            min_logvar = -2
            x_hat_var = max_logvar - torch.nn.functional.softplus(max_logvar - x_hat_var)
            x_hat_var = min_logvar + torch.nn.functional.softplus(x_hat_var - min_logvar)
        gnll = torch.nn.functional.gaussian_nll_loss(x_hat, x_target, x_hat_var.exp(), reduction='mean', full=False)
        mse = torch.nn.functional.mse_loss(x_hat, x_target, reduction='mean')
        if reconstruction_type == 'mse':
            return mse + vq_loss, mse, vq_loss
        elif reconstruction_type == 'gnll':
            return gnll + vq_loss, gnll, vq_loss

    @staticmethod
    def train(
        loss_fn,
        model,
        epochs,
        lr,
        dataset,
        device,
        file_name,
        overwrite = False,
        weight_decay = 0,
        batch_size = 256,
        target_dataset = None,
        reconstruction_type='mse',
        lim_logvar=False
    ):
        if len(file_name.split('/')) == 1:
            file_name = 'models/' + file_name
        if len(file_name.split('/')[-1].split('.')) == 1:
            file_name = file_name + '.pt'
        try:
            if overwrite: raise FileNotFoundError
            model = torch.load(file_name, map_location=device)
            print("Loaded saved model.")
        except FileNotFoundError:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            dataloader = [torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)]
            if target_dataset is not None:
                target_dataloader = torch.utils.data.DataLoader(dataset=target_dataset, batch_size=batch_size)
                dataloader.append(target_dataloader)
            for epoch in range(epochs):
                losses = []
                for data in zip(*dataloader):
                    optimizer.zero_grad()
                    if len(data) == 2:
                        x, target = data
                        x = x.to(device)
                        target = target.to(device)
                        x_hat, x_hat_var, _, vq_loss = model(x, device)
                        loss, _reconstruction, _vq_loss = loss_fn(target, x_hat, x_hat_var, vq_loss, reconstruction_type, lim_logvar)
                    elif len(data) == 1:
                        x = data[0]
                        x = x.to(device)
                        x_hat, x_hat_var, _, vq_loss = model(x, device)
                        loss, _reconstruction, _vq_loss = loss_fn(x, x_hat, x_hat_var, vq_loss, reconstruction_type, lim_logvar)
                    loss.backward()
                    losses.append([loss.cpu().detach(), _reconstruction.cpu().detach(), _vq_loss.cpu().detach()])
                    optimizer.step()
                epoch_loss = np.mean(losses, axis=0)
                print(f"Epoch: {epoch}: ",
                    f"| Reconstruction loss -> {epoch_loss[1]:.8f} ",
                    f"| VQ Loss -> {epoch_loss[2]:.8f} ",
                    f"| Total loss -> {epoch_loss[0]:.8f}",)
            os.makedirs('/'.join(file_name.split('/')[:-1]), exist_ok=True)
            torch.save(model, file_name)
        return model

    # def sample(self, num_samples, shape, device):
    #     latent_shape = (num_samples, shape[0] // self.reduction, shape[1] // self. reduction)
    #     ind = torch.randint(0, self.num_embeddings, latent_shape, device=device)
    #     with torch.no_grad():
    #         z = self.vq.quantize(ind)
    #         out = self.decoder(z)
    #     return out