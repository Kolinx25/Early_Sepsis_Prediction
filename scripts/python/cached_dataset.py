import numpy as np
import torch

from torch.utils.data import Dataset


class CachedTrajectoryDataset(Dataset):
    """
    Memory-mapped trajectory dataset.

    Returns:

        X:
            (12,69)

        y:
            scalar label

    """

    def __init__(self, split="train"):

        self.X = np.load(f"data/cache/{split}_X.npy", mmap_mode="r")

        self.y = np.load(f"data/cache/{split}_y.npy", mmap_mode="r")

        print(f"{split} cache loaded:", self.X.shape)

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        X = torch.tensor(self.X[index], dtype=torch.float32)

        y = torch.tensor(self.y[index], dtype=torch.float32)

        return X, y
