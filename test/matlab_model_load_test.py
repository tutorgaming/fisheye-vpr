import sys
sys.path.append("/Users/tutorgaming/Repository/workspace/fisheye-vpr")

# If exists
sys.path.append("/workspace/fisheye-vpr")

import torch
import numpy as np
from tqdm import tqdm
from dataloaders.isl2_3places_fisheye import ISL2_3Places_Fisheye_Dataset

from models.encoders.hloc_vgg16 import HLOCVGG16Encoder
from models.clustering.hloc_netvlad import HLOCNetVLAD

# TOrch Summary
import torchinfo

# class ModelLoadTest(object):
#     def __init__(self):
#         self.dataset = ISL2_3Places_Fisheye_Dataset()
#         self.encoder = HLOCVGG16Encoder()
#         self.clustering = HLOCNetVLAD()

class HLOC_VPR(torch.nn.Module):
    def __init__(self):
        super(HLOC_VPR, self).__init__()
        self.encoder = HLOCVGG16Encoder()
        self.clustering = HLOCNetVLAD()

    def forward(self, x):
        x = self.encoder(x)
        x = self.clustering(x)
        return x

if __name__ == "__main__":
    dataset = ISL2_3Places_Fisheye_Dataset()
    model = HLOC_VPR()
    torchinfo.summary(model, input_size=(1,3,256,256))
