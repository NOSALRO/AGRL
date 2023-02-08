import torch
import numpy as np
import matplotlib.pyplot as plt

datapoints  = np.load("random_data.npy")
vae = torch.load("../models/vae.pt")


reconstructed, _, _, _ = vae(torch.tensor(datapoints).float(), 'cpu', deterministic=True)
plt.scatter(x=datapoints[:,0], y=datapoints[:,1])
plt.scatter(x=reconstructed.detach().numpy()[:,0], y=reconstructed.detach().numpy()[:,1])
plt.show()

