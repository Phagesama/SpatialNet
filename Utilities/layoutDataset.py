import numpy as np
import torch
from torch.utils.data import Dataset

class layoutDataset(Dataset):
    def __init__(self, gen_para, *filename, device = 'cuda'):
        super().__init__()

        self.NofLinks = gen_para.NofLinks
        self.region_length = gen_para.region_length
        self.shortest_directLink_length = gen_para.shortest_directLink_length
        self.longest_directLink_length = gen_para.shortest_directLink_length
        self.setting_str = gen_para.setting_str
        self.cell_length = gen_para.cell_length
        self.n_grids = gen_para.n_grids

        if not len(filename):
            filename = [self.setting_str + "_txrx500.npy", self.setting_str + "_pathloss500.npy"]
        self.data = np.load(filename[0])
        self.NofSamples = len(self.data)
        self.pathloss_data = np.load(filename[1])

        self.tx_layout = torch.ones(self.NofSamples, self.NofLinks, 2, device=device)
        self.rx_layout = torch.ones(self.NofSamples, self.NofLinks, 2, device=device)
        self.pathloss = torch.zeros(self.NofSamples, self.NofLinks, self.NofLinks, device=device)
        for i in range(self.NofSamples):
            self.tx_layout[i, :, :] = torch.floor(torch.from_numpy(self.data[i][0]).to(device=device)/self.cell_length)
            self.rx_layout[i, :, :] = torch.floor(torch.from_numpy(self.data[i][1]).to(device=device)/self.cell_length)
            self.pathloss[i, :, :] = torch.from_numpy(self.pathloss_data[i]).to(device=device)

    def __getitem__(self, index):
        return self.tx_layout[index, :, :], self.rx_layout[index, :, :], self.pathloss[index, :, :]
    
    def __len__(self):
        return self.tx_layout.shape[0]
