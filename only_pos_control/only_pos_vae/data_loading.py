import re
import numpy as np
import torch
from torch.utils.data import Dataset

class StateData(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()

        self.__data = np.loadtxt(data_path, dtype=np.float32)

        self.__min = np.min(self.__data, axis=0)
        self.__max = np.max(self.__data, axis=0)
        self.__std = np.std(self.__data, axis=0)


    def __len__(self) -> int:
        return len(self.__data)

    def __getitem__(self, index) -> np.ndarray:
        return self.__data[index,:3]

    def scale(self, x) -> np.ndarray:
        return (x - self.__min) / (self.__max - self.__min)

    def get_data(self) -> np.ndarray:
        return self.__data[:, :3]