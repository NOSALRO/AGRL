from data_loading import StateData
import torch
from torch.distributions import MultivariateNormal
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = torch.load('models/vae.pt')

dataset = StateData("data/archive_10000.dat")

i = torch.tensor(dataset.get_data()).to(device)
decoder_mu, decoder_var, _, _ = vae.forward(i)
for mu, log_var, state in zip(decoder_mu, decoder_var, i):
    dist = MultivariateNormal(mu, torch.diag(torch.exp(0.5 * log_var)))
    for sample in i:
        print(np.linalg.norm(state.cpu() - sample.cpu().detach().numpy(), ord=2), '->', dist.log_prob(sample).cpu().detach().numpy())
