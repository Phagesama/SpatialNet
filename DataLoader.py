import numpy as np
import torch
from torch.utils.data import Dataset

class layoutDataset(Dataset):
    def __init__(self, filename):
        super().__init__()
        self.data = np.load(filename)
        self.NofSamples = len(self.data)
        self.NofLinks = len(self.data[0][0])
        for sample in self.data:
            pass