import torch

from SampleGeneration import SampleGeneration
from general_parameters import parameters
from DataLoader import layoutDataset

if __name__ == "__main__":

    gen_para = parameters()
    sample_generator = SampleGeneration(gen_para)
    sample_generator.trainSamples(800)

    layouts = layoutDataset(gen_para)
    print(layouts[0])
    print(gen_para.setting_str + "generate successfully!")


