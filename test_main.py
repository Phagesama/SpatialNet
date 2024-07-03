import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from Utilities.general_parameters import parameters
from Utilities.layoutDataset import layoutDataset
from Neural_Network_Model.SpatialNet import SpatialNet
from Neural_Network_Model.Loss import Objective_Func

if __name__ == "__main__":

    losses = torch.load("losses.pt")
    plt.plot(range(1000), losses)
    plt.savefig("loss.png")

    gen_para = parameters()

    layouts = layoutDataset(gen_para)
    batch_size=2
    train_dataloader = DataLoader(layouts, batch_size=batch_size)

    spatial_net = SpatialNet(gen_para.NofBlocks, 63, gen_para).to(device="cuda")
    spatial_net.load_state_dict(torch.load("checkpoint_epoch800.pt"))

    kernel = spatial_net.conv_kernel[0, :, :]
    kernel = (kernel - torch.min(kernel))/(torch.max(kernel)-torch.min(kernel))
    plt.imshow(kernel.tolist(), cmap=plt.cm.gray)
    plt.savefig("kernel_0.png")
    
    loss_fun = Objective_Func(gen_para, 4, 10e5, 10e2)

    learning_rate = 3e-5
    optimizer = torch.optim.SGD(spatial_net.parameters(), lr=learning_rate)

    for tx_layout, rx_layout, pathloss in train_dataloader:
        T = torch.cat([torch.tensor(range(batch_size), device="cuda").expand(gen_para.NofBlocks, gen_para.NofLinks, batch_size).permute(2, 0, 1).unsqueeze(3),
                        torch.tensor(range(gen_para.NofBlocks), device="cuda").expand(batch_size, gen_para.NofLinks, gen_para.NofBlocks).permute(0, 2, 1).unsqueeze(3), 
                           tx_layout.expand(gen_para.NofBlocks, batch_size, gen_para.NofLinks, 2).permute(1, 0, 2, 3)], dim=3).to(dtype=torch.double)
        R = torch.cat([torch.tensor(range(batch_size), device="cuda").expand(gen_para.NofBlocks, gen_para.NofLinks, batch_size).permute(2, 0, 1).unsqueeze(3),
                        torch.tensor(range(gen_para.NofBlocks), device="cuda").expand(batch_size, gen_para.NofLinks, gen_para.NofBlocks).permute(0, 2, 1).unsqueeze(3), 
                           rx_layout.expand(gen_para.NofBlocks, batch_size, gen_para.NofLinks, 2).permute(1, 0, 2, 3)], dim=3).to(dtype=torch.double)
            
        powers = spatial_net(T, R, torch.ones(batch_size, gen_para.NofBlocks, gen_para.NofLinks, device="cuda", dtype=torch.double))
        print(powers)
        outa = loss_fun.outage(pathloss, powers)
        #print(outa[0, -1, :])
        break
    