import torch
from torch.utils.data import Dataset

# TODO: Define dataset classes.

class MyDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.data = torch.randn(100, 10)
        self.targets = torch.randn(100, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y