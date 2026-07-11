"""
PyTorch Dataset wrapping the processed DREAMER windows + binary stress labels.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class EEGWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        X: (N, T, C) float32
        y: (N,) int64
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
