import torch
import torch.nn as nn

class SpatialNet(nn.Module):

    def __init__(self, N, C, gen_para):
        super().__init__()

        self.NofLinks = gen_para.NofLinks
        self.N = N
        self.C = C

        kernel = torch.randn(N, C, C)
        self.conv_tx = nn.Parameter(data=kernel)
        self.conv_rx = nn.Parameter(data=kernel)

        self.first = nn.Linear(4*N*self.NofLinks + 2*N, N*self.NofLinks)

    def forward(self, TxINT, RxINT, powers):
        for i in range(self.NofLinks):
            TxIDX = TxINT
            TxIDX = TxINT[:, 0] - TxINT[i, 0] + (self.C + 1)/2
            TxINT_dense = torch.sparse_coo_tensor(TxINT[:, :1].T, TxINT[:, 2] * )
        for 
        return self.first(torch.cat([TxINT.unsqueeze(0), RxINT.unsqueeze(0)], dim=0))