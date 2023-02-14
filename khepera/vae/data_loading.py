import torch
import numpy as np

class ObstacleData(torch.utils.data.Dataset):

    def __init__(self, path):
        super().__init__()
        self.__data = np.loadtxt(path, dtype=np.float32)

    def scale_data(self):
        __min = np.min(self.__data)
        __max = np.max(self.__data)
        self.__data = (self.__data - __min) / ( __max - __min)
        return __min, __max

    def get_data(self):
        return self.__data

    def __len__(self):
        return len(self.__data)

    def __getitem__(self, idx):
        return self.__data[idx]