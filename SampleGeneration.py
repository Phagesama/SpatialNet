"""
Generate train samples of SpatialNet
"""
import numpy as np

class SampleGeneration:
    def __init__(self, NofSamples, NofLinks, region_length) -> None:
        '''
            region_length: the length and width of the region where D2D links locates
        '''
        self.NofSamples = NofSamples
        self.NofLinks = NofLinks
        self.region_length = region_length

    def generateOneSample(self):
        tx_locat = np.random.uniform(low=0, high=self.region_length, size=[self.NofLinks, 2])
        d_min = np.random.uniform(low=2, high=70)
        d_max = np.random.uniform(low=d_min, high=70)
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
        
        return tx_locat, rx_locat

    def trainSamples(self):
        pass
