import torch

from Utilities.SampleGeneration import SampleGeneration
from Utilities.general_parameters import parameters
from Utilities.layoutDataset import layoutDataset

if __name__ == "__main__":

    gen_para = parameters()
    sample_generator = SampleGeneration(gen_para)
    sample_generator.trainSamples(500)

    layouts = layoutDataset(gen_para)
    print(layouts[0])
    print(gen_para.setting_str + "generate successfully!")


