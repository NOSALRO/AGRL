from data_loading import StateData
import torch
from torch.distributions import Normal
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = torch.load('models/vae.pt')

dataset = StateData("data/archive_10000.dat")

i = torch.tensor(dataset.get_data()).to(device)
decoder_mu, decoder_var, _, _ = vae.forward(i)
for mu, log_var, state in zip(decoder_mu, decoder_var, i):
    dist = Normal(mu, torch.exp(0.5 * log_var))
    _, samples, _ = vae.sample(10, device)
    for sample in samples:
        print("Next state")
        print(f"distance {np.linalg.norm(state.cpu() - sample.cpu().detach().numpy(), ord=2)}")
        print(f"reward {np.sum(dist.log_prob(sample).cpu().detach().numpy())}")