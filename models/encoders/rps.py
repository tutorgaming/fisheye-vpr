#!/usr/bin/env python3
"""
RPS Encoder Class
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

from torchvision.models import resnet18

#####################################################################
# Class
#####################################################################
class Resnet18Encoder(nn.Module):
    def __init__(self, fine_tuning=False):
        # Config
        self.name = "rps_encoder"
        self.fine_tuning = fine_tuning


    def forward(self, x):
        return x
