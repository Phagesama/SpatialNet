import torch
import torch.nn as nn
import torch.nn.functional as F

class Objective_Func(nn.Module):
    def __init__(self, pathloss, gen_para, t, lam, rho, device = "cuda"):
        super().__init__()
        self.pathloss = pathloss
        self.NofLinks = pathloss.shape[0]
        self.noise_power = gen_para.input_noise_power
        self.dt = gen_para.dt
        self.device = device
        self.pow2t_array = torch.pow(2, torch.arange(0, t, self.dt, device=device))
        self.pow2t_1_array = self.pow2t_array - 1
        self.t_array_len = self.t_array.shape[0]
        self.outa = None
        self.lam = lam
        self.rho = rho
    
    def outage(self, powers):
        NofBlocks = powers.shape[0]
        signal = self.pathloss.expand(NofBlocks, self.NofLinks, self.NofLinks) * powers.expand(self.NofLinks, NofBlocks, self.NofLinks).permute(1, 0, 2)
        y_ii = torch.diag(signal)
        y_ij = torch.sum(signal - y_ii, dim=2)
        sinr = y_ii/(y_ij + self.noise_power)

        outa = []
        Q_i1 = 1 - torch.exp(-(self.pow2t_1_array.expand(self.NofLinks, self.t_array_len) * sinr[0, :].expand(self.NofLinks, self.t_array_len)))
        outa.append(Q_i1[:, self.t_array_len])
        for n in range(1, NofBlocks):
            q_ii = torch.exp(-(self.pow2t_1_array.expand(self.NofLinks, self.t_array_len) * sinr[n, :].expand(self.NofLinks, self.t_array_len))) * self.pow2t_array.expand(self.NofLinks, self.t_array_len) * sinr[n, :].expand(self.NofLinks, self.t_array_len) * torch.log(2)
            Q_in = F.conv1d(Q_i1, q_ii.unsqueeze(1), padding=self.t_array_len - 1, groups=self.NofLinks)[:, :self.t_array_len] * self.dt
            outa.append(Q_in[:, self.t_array_len])
        
        return torch.stack(outa)

    def forward(self, powers):

        NofBlocks = powers.shape[0]
        self.outa = self.outage(powers)

        E = powers[0, :] + torch.sum(powers[1:, :] * self.outa[:NofBlocks - 1, :], dim=0)

        D = 1 + torch.sum(self.outa[:NofBlocks - 1, :], dim=0)

        loss = torch.sum(E) + self.lam*torch.sum(self.outa[NofBlocks,:]) + self.rho*torch.sum(D)

        return loss
