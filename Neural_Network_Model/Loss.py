import torch
import torch.nn as nn

class Objective_Func(nn.Module):
    def __init__(self, pathloss, gen_para, t, device = "cuda"):
        super().__init__()
        self.pathloss = pathloss
        self.NofLinks = pathloss.shape[0]
        self.noise_power = gen_para.input_noise_power
        self.device = device
        self.t_array = torch.arange(0, t, gen_para.dt, device=device)

    def forward(self, powers):

        NofBlocks = powers.shape[0]
        signal = self.pathloss.expand(NofBlocks, self.NofLinks, self.NofLinks) * powers.expand(self.NofLinks, NofBlocks, self.NofLinks).permute(1, 0, 2)
        y_ii = torch.diag(signal)
        y_ij = torch.sum(signal - y_ii, dim=2)
        sinr = y_ii/(y_ij + self.noise_power)
        for i in range(NofBlocks):
            pass
        pass