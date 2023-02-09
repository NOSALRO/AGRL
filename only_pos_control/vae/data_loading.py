import re
import numpy as np
import torch
from torch.utils.data import Dataset

class StateData(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()

        self.__data = np.loadtxt(data_path, dtype=np.float32)
        self.min = self.__data.min()
        self.max = self.__data.max()

    def __len__(self) -> int:
        return len(self.__data)

    def __getitem__(self, index) -> np.ndarray:
        return self.__data[index,:3]

    def scale(self, x) -> np.ndarray:
        self.__data = (x - self.min) / (self.max - self.min)

    def get_data(self) -> np.ndarray:
        return self.__data[:, :3]