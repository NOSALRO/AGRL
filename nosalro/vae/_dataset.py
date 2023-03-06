import copy
import torch
import numpy as np


class StatesDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            path,
            *,
            angle_to_sin_cos = False,
            angle_column = 0,
            columns_select = None
        ):
        super().__init__()
        self.states = np.loadtxt(path, dtype=np.float32)
        self.columns_select = columns_select if columns_select else np.arange(self.states.shape[-1])
        self.unscaled_data = None

        if angle_to_sin_cos and angle_column in self.columns_select:
            _sines = np.sin(self.states[:,angle_column]).reshape(-1,1)
            _cosines = np.cos(self.states[:,angle_column]).reshape(-1,1)
            self.states = self.states[:, self.columns_select]
            self.states = np.concatenate((self.states, _cosines, _sines), axis=1)
            self.states = np.delete(self.states, angle_column, axis=1)
        elif not angle_to_sin_cos or angle_column not in self.columns_select:
            self.states = self.states[:, self.columns_select]

    def scale_data(self, scaler):
        # TODO: Better scaling implementation to avoid keeping 2 copies of the dataset.
        self.unscaled_data = copy.deepcopy(self.states)
        self.states = scaler(self.states)

    def get_data(self):
        return self.unscaled_data if self.unscaled_data is not None else self.states

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx]