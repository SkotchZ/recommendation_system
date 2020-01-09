from torch.utils.data import Dataset
import torch

class CustomDataset(Dataset):
    def __init__(self, x):
        self.x = torch.Tensor(x)

    def __len__(self):
        self.len = len(self.x)
        return self.len

    def __getitem__(self, index):
        return self.x[index]
