import matplotlib.pyplot as plt
import numpy as np
import torch

def vizualization(data):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    state_space = ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=data[:,2], cmap='coolwarm')
    fig.colorbar(state_space, shrink=0.5, aspect=5)
    plt.show()



def train(model, epochs, dataloader):

    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        l = []
        for x in dataloader:

            optimizer.zero_grad()
            x_hat = model(x)
            loss = ((x - x_hat)**2).sum() + model.encoder.kl
            loss.backward()
            l.append(loss.detach())
            optimizer.step()

        print(f"Epoch: {epoch}, train_loss = {np.mean(l)}")
    return model


def plot_latent(autoencoder, data):
    zs = []
    for x in data:
        z = autoencoder.encoder(x)
        for i in z.detach().numpy():
            zs.append(i)
    
    zs = np.array(zs)
    vizualization(zs)

def plot_reconstructed(autoencoder, data):
    zs = []
    for x in data:
        z = autoencoder(x)
        for i in z.detach().numpy():
            zs.append(i)
            
    zs = np.array(zs)
    vizualization(zs)
