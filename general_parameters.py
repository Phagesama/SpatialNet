# General parameters

import numpy as np

class parameters():
    def __init__(self):
        # wireless network settings
        self.NofLinks = 50
        self.region_length = 1000
        self.shortest_directLink_length = 2
        self.longest_directLink_length = 70
        self.shortest_crossLink_length = 1
        self.bandwidth = 5e6
        self.carrier_f = 2.4e9
        self.tx_height = 1.5
        self.rx_height = 1.5
        self.antenna_gain_decibel = 2.5
        self.tx_power_milli_decibel = 40
        self.tx_power = np.power(10, (self.tx_power_milli_decibel-30)/10)
        self.noise_density_milli_decibel = -169
        self.input_noise_power = np.power(10, ((self.noise_density_milli_decibel-30)/10)) * self.bandwidth
        self.output_noise_power = self.input_noise_power
        self.SNR_gap_dB = 6
        self.SNR_gap = np.power(10, self.SNR_gap_dB/10)
        self.setting_str = "{}_links_{}X{}_{}_{}_length".format(self.NofLinks, self.region_length, self.region_length, self.shortest_directLink_length, self.longest_directLink_length)
        # 2D occupancy grid setting
        self.cell_length = 5
        self.n_grids = np.round(self.field_length/self.cell_length).astype(int)