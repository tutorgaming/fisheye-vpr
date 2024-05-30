#!/usr/bin/env python3
"""
VGG-16 Encoder Class
- To Extract Local Feature Patch from Images
"""
# Author : Theppasith N. <tutorgaming@gmail.com>
# Date : 04-Jun-2023
#####################################################################
# Imports
#####################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

# Matlab Layer weights
from scipy.io import loadmat

#####################################################################
# Class
#####################################################################
class VGG16Encoder(nn.Module):
    def __init__(self, fine_tuning=False):
        super(VGG16Encoder, self).__init__()
        # Config
        self.name = "vgg16"
        self.fine_tuning = fine_tuning
        # Load Pretrained Model
        encoder = vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1
        )
        # Assemble model
        self.feature_extractor = nn.Sequential(
            encoder.features[0]
        )
        # Freeze Weight or Training more ?
        if self.fine_tuning:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        last_dim_size = list(self.feature_extractor.parameters())[-1].shape[0]
        print("[VGG16] Output Dim Size: {}".format(last_dim_size))

    def assign_matlab_weights(self, mat_weight):
        """
        Given MATLAB weights, assign them to the corresponding layers
        """
        self.feature_extractor




    def forward(self, x):
        return self.feature_extractor(x)
