import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import h5py
import numpy as np


class TrafficDataset(Dataset):

    def __init__(self, path, seq_len=6, pred_len=2, max_samples=8000):

        with h5py.File(path, "r") as f:
            data = f["df"]["block0_values"][:]

        data = data[:max_samples]

        self.scaler = StandardScaler()
        self.data = self.scaler.fit_transform(data)

        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, idx):

        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]

        return (
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y[:, 0], dtype=torch.float32)
        )