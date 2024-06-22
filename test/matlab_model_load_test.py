import sys
sys.path.append("/Users/tutorgaming/Repository/workspace/fisheye-vpr")

import torch
import numpy as np
from tqdm import tqdm
from dataloaders.isl2_3places_fisheye import ISL2_3Places_Fisheye_Dataset

from models.encoders.hloc_vgg16 import HLOCVGG16Encoder
from models.clustering.hloc_netvlad import HLOCNetVLAD

class ModelLoadTest(object):
    def __init__(self):
        self.dataset = ISL2_3Places_Fisheye_Dataset()
        self.encoder = HLOCVGG16Encoder()
        self.clustering = HLOCNetVLAD()

if __name__ == "__main__":
    ModelLoadTest()