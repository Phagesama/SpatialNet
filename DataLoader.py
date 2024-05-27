import numpy as np
import torch
from torch.utils.data import Dataset

class layoutDataset(Dataset):
    def __init__(self, gen_para, *filename, device = 'cuda'):
        super().__init__()

        self.NofSamples = len(self.data)
        self.NofLinks = gen_para.NofLinks
        self.region_length = gen_para.region_length
        self.shortest_directLink_length = gen_para.shortest_directLink_length
        self.longest_directLink_length = gen_para.shortest_directLink_length
        self.setting_str = gen_para.setting_str
        self.cell_length = gen_para.cell_length
        self.n_grids = gen_para.n_grids

        if not len(filename):
            filename = self.setting_str + ".npy" 
        self.data = np.load(filename)

        self.tx_layout = torch.ones(self.NofSamples, self.n_grids, 3, device=device)
        self.rx_layout = torch.ones(self.NofSamples, self.n_grids, 3, device=device)
        self.pathloss = torch.zeros(self.NofSamples, self.NofLinks, self.NofLinks, device=device)
        for i in range(self.NofSamples):
            self.tx_layout[i, :, 1:2] = torch.floor(torch.from_numpy(self.data[i][0], device=device)/self.cell_length)
            self.rx_layout[i, :, 1:2] = torch.floor(torch.from_numpy(self.data[i][1], device=device)/self.cell_length)
            self.pathloss[i, :, :] = torch.from_numpy(self.data[i][2], device=device)

    def __getitem__(self, index):
        return torch.cat(self.tx_layout[index, :, :].unsqueeze(0), self.rx_layout[index, :, :].unsqueeze(0), dim=0)
