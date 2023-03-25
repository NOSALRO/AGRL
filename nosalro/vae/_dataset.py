import copy
import torch
import numpy as np


class StatesDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            x = None,
            path = None,
            *,
            transforms = None
        ):
        assert x is not None or path is not None, "path or data should be provided."
        super().__init__()
        if path is not None:
            self.states = np.loadtxt(path, dtype=np.float32)
        elif x is not None:
            self.states = x.astype(np.float32)
        if transforms is not None:
            self.transforms = transforms
            self.states = transforms(self.states)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx]

    def set_data(self, data):
        self.states = data

    def inverse(self, inverse_ops = []):
        self.states = self.transforms.inverse(self.states, inverse_ops)