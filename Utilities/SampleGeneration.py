# Generate train samples of SpatialNet

import numpy as np
from Utilities.helper_functions import computePathloss

class SampleGeneration:
    def __init__(self, gen_para) -> None:
        '''
            region_length: the length and width of the region where D2D links locates
        '''
        self.NofLinks = gen_para.NofLinks
        self.region_length = gen_para.region_length
        self.shortest_directLink_length = gen_para.shortest_directLink_length
        self.longest_directLink_length = gen_para.shortest_directLink_length
        self.setting_str = gen_para.setting_str
        self.gen_para = gen_para

    def generateOneSample(self):
        tx_locat = np.random.uniform(low=0, high=self.region_length, size=[self.NofLinks, 2])
        d_min = np.random.uniform(low=self.shortest_directLink_length, high=self.longest_directLink_length)
        d_max = np.random.uniform(low=d_min, high=self.longest_directLink_length)
        rx_locat = np.copy(tx_locat)
        for i in range(self.NofLinks):
            got_valid_rx = False
            while (not got_valid_rx):
                rx_dist = np.random.uniform(low=d_min, high=d_max)
                rx_direct = np.random.uniform(low=0, high=2*np.pi)
                rx_locat[i, 0] = tx_locat[i, 0] + np.cos(rx_direct) * rx_dist
                rx_locat[i, 1] = tx_locat[i, 1] + np.sin(rx_direct) * rx_dist
                if (0<=rx_locat[i, 0]<=self.region_length and 0<=rx_locat[i, 1]<=self.region_length):
                    got_valid_rx = True
        
        distance = np.zeros([self.NofLinks, self.NofLinks])
        for i in range(self.NofLinks):
            for j in range(self.NofLinks):
                distance[i, j] = np.sqrt((rx_locat[i, 0]-tx_locat[j, 0])**2 + (rx_locat[i, 1]-tx_locat[j, 1])**2)
        pathloss = computePathloss(self.gen_para, distance)

        return tx_locat, rx_locat, pathloss

    def trainSamples(self, NofSamples:int, *filename):
        if not len(filename):
            filename = [self.setting_str + "_txrx.npy", self.setting_str + "_pathloss.npy"]
        txrx_samples = []
        pathloss_samples = []
        for i in range(NofSamples):
            tx_locat, rx_locat, pathloss = self.generateOneSample()
            txrx_samples.append([tx_locat, rx_locat])
            pathloss_samples.append(pathloss)

        np.save(filename[0], txrx_samples)
        np.save(filename[1], pathloss_samples)
