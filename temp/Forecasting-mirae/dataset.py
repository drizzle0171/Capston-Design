import torch
import numpy as np
from torch.utils.data import Dataset

class MiraeDataset(Dataset):
    def __init__(self, x, y, max_x=None, min_x=None, is_train=True):
        self.x, self.y = x, y
        if is_train:
            self.max_x = np.max(x)
            self.min_x = np.min(x)
        else:
            self.max_x = max_x
            self.min_x = min_x
        
        self.x = (self.x - self.min_x) / (self.max_x - self.min_x)
        self.y = (self.y - self.min_x) / (self.max_x - self.min_x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]).float(), torch.from_numpy(self.y[idx]).float()
