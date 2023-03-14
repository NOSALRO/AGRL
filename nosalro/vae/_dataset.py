import copy
import torch
import numpy as np


class StatesDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            x = None,
            path = None,
            *,
            angle_to_sin_cos = False,
            angle_column = 0,
            columns_select = None
        ):
        assert x is not None or path is not None, "path or data should be provided."
        super().__init__()
        if path is not None:
            self.states = np.loadtxt(path, dtype=np.float32)
        elif x is not None:
            self.states = x.astype(np.float32)
        self.columns_select = columns_select if columns_select else np.arange(self.states.shape[-1])
        self.scaled = False

        if angle_to_sin_cos and angle_column in self.columns_select:
            _sines = np.sin(self.states[:,angle_column]).reshape(-1,1)
            _cosines = np.cos(self.states[:,angle_column]).reshape(-1,1)
            self.states = self.states[:, self.columns_select]
            self.states = np.concatenate((self.states, _cosines, _sines), axis=1)
            self.states = np.delete(self.states, angle_column, axis=1)
        elif not angle_to_sin_cos or angle_column not in self.columns_select:
            self.states = self.states[:, self.columns_select]

    def scale_data(self, scaler):
        self.scaler = scaler
        self.states = self.scaler(self.states)
        self.scaled = True

    def get_data(self, unscaled=False):
        if unscaled and self.scaled:
            return self.scaler(self.states, True)
        else:
            return self.states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx]