import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Objective_Func(nn.Module):
    def __init__(self, gen_para, t, lam, rho, device = "cuda"):
        super().__init__()
        self.NofLinks = gen_para.NofLinks
        self.noise_power = gen_para.input_noise_power
        self.dt = gen_para.dt
        self.device = device
        self.pow2t_array = torch.pow(2, torch.arange(0, t+0.1*self.dt, self.dt, device=device))
        self.pow2t_1_array = self.pow2t_array - 1
        self.t_array_len = self.pow2t_array.shape[0]
        self.outa = None
        self.lam = lam
        self.rho = rho
        self.power_level = gen_para.tx_power
    
    def outage(self, pathloss, powers):

        batch_size = powers.shape[0]
        NofBlocks = powers.shape[1]
        signal = pathloss.expand(NofBlocks, batch_size, self.NofLinks, self.NofLinks).permute(1, 0, 2 ,3) * powers.expand(self.NofLinks, batch_size, NofBlocks, self.NofLinks).permute(1, 2, 3, 0)
        y_ii = torch.diagonal(signal, dim1=2, dim2=3)
        y_ij = torch.sum(signal, dim=3) - y_ii
        sinr = y_ii/(y_ij + self.noise_power)
        #print(sinr)
        #sinr.clamp_(1e-169)

        outa = []
        Q_in = (1 - torch.exp(-(self.pow2t_1_array.expand(batch_size, self.NofLinks, self.t_array_len) / sinr[:, 0, :].expand(self.t_array_len, batch_size, self.NofLinks).permute(1, 2, 0))))[:, :, 1:]
        outa.append(Q_in[:, :, self.t_array_len - 2])
        Q_in = Q_in.reshape(batch_size * self.NofLinks, self.t_array_len-1)

        for n in range(1, NofBlocks):
            F_in = (1 - torch.exp(-(self.pow2t_1_array.expand(batch_size, self.NofLinks, self.t_array_len) / sinr[:, 0, :].expand(self.t_array_len, batch_size, self.NofLinks).permute(1, 2, 0)))).reshape(batch_size * self.NofLinks, self.t_array_len)
            f_in = F_in.diff().flip(dims=[1])/self.dt
            #q_in = torch.exp(-(self.pow2t_1_array.expand(batch_size, self.NofLinks, self.t_array_len) / sinr[:, n, :].expand(self.t_array_len, batch_size, self.NofLinks).permute(1, 2, 0))) * self.pow2t_array.expand(batch_size, self.NofLinks, self.t_array_len) / sinr[:, n, :].expand(self.t_array_len, batch_size, self.NofLinks).permute(1, 2, 0) * np.log(2)
            #q_in = q_in.reshape(batch_size * self.NofLinks, self.t_array_len).flip(dims=[1])
            Q_in = F.conv1d(Q_in, f_in.unsqueeze(1), padding=self.t_array_len - 2, groups=batch_size*self.NofLinks)[:, :self.t_array_len - 1] * self.dt
            outa.append(Q_in.view(batch_size, self.NofLinks, self.t_array_len-1)[:, :, self.t_array_len - 2])
        
        return torch.stack(outa, dim=1)

    def forward(self, pathloss, powers):

        power = powers * self.power_level
        NofBlocks = power.shape[1]
        self.outa = self.outage(pathloss, power)
        #print(self.outa[0,:,0])

        E = power[:, 0, :] + torch.sum(power[:, 1:, :] * self.outa[:, :NofBlocks - 1, :], dim=1)

        D = 1 + torch.sum(self.outa[:, :NofBlocks - 1, :], dim=1)

        loss = torch.sum(E) + self.lam*torch.sum(self.outa[:, NofBlocks - 1, :]) + self.rho*torch.sum(D)

        return loss
