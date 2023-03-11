import torch
from torch.utils.data import Dataset
import numpy as np
class AHBDataset(Dataset):
    def __init__(self,AH_dataset,BH_dataset):
        self.AH=AH_dataset
        self.BH=BH_dataset
    def __len__(self):
        return np.array(self.AH).shape[0]
    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.AH[idx])),torch.from_numpy(np.array(self.BH[idx]))
