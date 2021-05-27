import torch
from torch.utils.data import Dataset
import numpy as np


class Dataset(Dataset):

    def __init__(self, X, Y):
        self.X = torch.tensor(np.array(X)).float()
        self.Y = torch.tensor(Y.values).int().type(torch.LongTensor)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class TargetDataset(Dataset):

    def __init__(self, X):

        self.X = torch.tensor(np.array(X)).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], 1