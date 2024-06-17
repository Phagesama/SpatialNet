import torch
import torch.nn as nn

class SpatialNet(nn.Module):

    def __init__(self, N, C, gen_para):
        super().__init__()

        self.NofLinks = gen_para.NofLinks
        self.N = N
        self.C = C

        kernel = torch.ones(N, C, C, device="cuda", dtype=torch.double)
        self.conv_kernel = nn.Parameter(data=kernel)
        self.normalize_layer = nn.BatchNorm2d(1)
        self.deep_network = nn.Sequential(
            nn.Linear(4 * self.N * self.NofLinks + 2 * self.N, 4 * self.N * self.NofLinks + 2 * self.N * 30, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(4 * self.N * self.NofLinks + 2 * self.N * 30, 4 * self.N * self.NofLinks + 2 * self.N * 30, dtype=torch.double),
            nn.ReLU(),
            nn.Linear(4 * self.N * self.NofLinks + 2 * self.N * 30, self.N * self.NofLinks, dtype=torch.double),
            nn.Sigmoid()
        )

    def forward(self, T, R, powers):
        """
        TxINT: Sparsed Matrix: batch_size, NofBLocks, NofLinks, [batch_idx, block_idx, grid_x, grid_y]
        RxINT: The same as TxINT
        powers: block_idx, link_idx
        """
        batch_size = T.shape[0]
        device = T.device
        Tx_INT = []
        Rx_INT = []
        DSS = []
        for i in range(self.NofLinks):

            TxIDX = T.clone()
            RxIDX = R.clone()
            DSS.append(self.conv_kernel.expand(batch_size, self.N, self.C, self.C).reshape(batch_size*self.N, self.C, self.C)[torch.tensor(range(batch_size*self.N), device=device), 
                                                                                                                              (R[:, :, i, 2].view(batch_size*self.N) - T[:, :, i, 2].view(batch_size*self.N) + (self.C + 1)/2).to(dtype = torch.int),  
                                                                                                                              (R[:, :, i, 3].view(batch_size*self.N) - T[:, :, i, 3].view(batch_size*self.N) + (self.C + 1)/2).to(dtype = torch.int)].view(batch_size, self.N, 1))
            
            TxIDX[:, :, :, 2] = TxIDX[:, :, :, 2] - RxIDX[:, :, i, 2].unsqueeze(2) + (self.C + 1)/2
            TxIDX[:, :, :, 3] = TxIDX[:, :, :, 3] - RxIDX[:, :, i, 3].unsqueeze(2) + (self.C + 1)/2
            TxIDX = TxIDX.view(batch_size * self.N * self.NofLinks, 4).permute(1, 0)
            value = powers.view(batch_size * self.N * self.NofLinks)[(TxIDX[2, :]>=0) & (TxIDX[2, :]<=62) & (TxIDX[3, :]>=0) & (TxIDX[3,:]<=62)]
            TxIDX = TxIDX[:, (TxIDX[2, :]>=0) & (TxIDX[2, :]<=62) & (TxIDX[3, :]>=0) & (TxIDX[3,:]<=62)]
            Tx_Conv = torch.sparse_coo_tensor(TxIDX, value, size=(batch_size, self.N, self.C, self.C)).to_dense() * self.conv_kernel.expand(batch_size, self.N, self.C, self.C)
            Tx_INT.append(torch.sum(torch.sum(Tx_Conv, dim=3), dim=2, keepdim=True))

            TxIDX = T.clone()
            RxIDX = R.clone()
            
            RxIDX[:, :, :, 2] = RxIDX[:, :, :, 2] - TxIDX[:, :, i, 2].unsqueeze(2) + (self.C + 1)/2
            RxIDX[:, :, :, 3] = RxIDX[:, :, :, 3] - TxIDX[:, :, i, 3].unsqueeze(2) + (self.C + 1)/2
            RxIDX = RxIDX.view(batch_size * self.N * self.NofLinks, 4).permute(1, 0)
            value = powers.view(batch_size * self.N * self.NofLinks)[(RxIDX[2, :]>=0) & (RxIDX[2, :]<=62) & (RxIDX[3, :]>=0) & (RxIDX[3,:]<=62)]
            RxIDX = RxIDX[:, (RxIDX[2, :]>=0) & (RxIDX[2, :]<=62) & (RxIDX[3, :]>=0) & (RxIDX[3,:]<=62)]
            Rx_Conv = torch.sparse_coo_tensor(RxIDX, value, size=(batch_size, self.N, self.C, self.C)).to_dense() * self.conv_kernel.expand(batch_size, self.N, self.C, self.C)
            Rx_INT.append(torch.sum(torch.sum(Rx_Conv, dim=3), dim=2, keepdim=True))

        Tx_INT = torch.stack(Tx_INT, dim=2) #batch_size, NofBlocks, NofLinks
        Rx_INT = torch.stack(Rx_INT, dim=2) #batch_size, NofBlocks, NofLinks
        DSS = torch.stack(DSS, dim=2) #batch_size, NofBlocks, NofLinks
        deep_input = nn.functional.normalize(torch.cat([Tx_INT.reshape(batch_size, self.N * self.NofLinks), Rx_INT.reshape(batch_size, self.N * self.NofLinks), DSS.reshape(batch_size, self.N * self.NofLinks)], dim=1).reshape(batch_size * 3 * self.N * self.NofLinks), p=2, dim=0).reshape(batch_size, 3 * self.N * self.NofLinks)

        DSS_min = torch.min(deep_input[:, 2*self.N*self.NofLinks:].reshape(batch_size, self.N,  self.NofLinks), dim=2, keepdim=False).values.squeeze() #batch_size, NofBlocks
        DSS_max = torch.max(deep_input[:, 2*self.N*self.NofLinks:].reshape(batch_size, self.N,  self.NofLinks), dim=2).values.squeeze() #batch_size, NofBlocks
        deep_input = torch.cat([deep_input.squeeze(), DSS_min, DSS_max, powers.reshape(batch_size, self.N * self.NofLinks)], dim=1)


        return self.deep_network(deep_input).reshape(batch_size, self.N, self.NofLinks) * (1 - 1e-169) + 1e-169