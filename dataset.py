import h5py
import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    def __init__(self, path, load_into_memory=True,normalize= True,transform=None):
        self.path = path
        self.data = h5py.File(path, "r")
        self.load_into_memory = load_into_memory
        self.transform = transform
        if load_into_memory:
            self.inputs = np.stack([self.data["density"], self.data["recording_date"]], axis=1)
            self.labels = self.data["labels"][:]


    def __len__(self):
        if self.load_into_memory:
            return self.inputs.shape[0]
        else:
            return self.data["density"].shape[0]

    def __getitem__(self, idx):
        if self.load_into_memory:
            x, y = np.float32(self.inputs[idx]), np.float32(self.labels[idx])
        else:
            x = np.stack([self.data["density"][idx], self.data["recording_date"][idx]], axis=1)
            y = np.float32(self.data["labels"][idx])
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        
        if self.transform:
            x = self.transform(x)
        
        return x, y