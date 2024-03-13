#!/usr/bin/env python3
"""
RESNET-18 Encoder Class
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

from torchvision.models import resnet18, ResNet18_Weights

#####################################################################
# Class
#####################################################################
class Resnet18Encoder(nn.Module):
    def __init__(self, fine_tuning=False):
        super(Resnet18Encoder, self).__init__()
        # Config
        self.name = "resnet18"
        self.fine_tuning = fine_tuning
        # Load Pretrained Model
        encoder = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1, # Pretrained with this weight
            # pretrained=True
        )

        # Assemble model
        self.feature_extractor = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,
            encoder.layer2,
            encoder.layer3,
            encoder.layer4,
        )

        # Freeze Weight or Training more ?
        if self.fine_tuning:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        last_dim_size = list(self.feature_extractor.parameters())[-1].shape[0]
        print("[RESNET18] Output Dim Size: {}".format(last_dim_size))

    def forward(self, x):
        return self.feature_extractor(x)

    def get_last_dim(self):
        last_dim_size = list(self.feature_extractor.parameters())[-1].shape[0]
        return last_dim_size