#!/usr/bin/env python3
"""
Visual Place Recognition Model
"""
# Author : Theppasith N. <tutorgaming@gmail.com>
# Date : 04-Jun-2023
#####################################################################
# Imports
#####################################################################
import torch
import torch.nn as nn

#####################################################################
# Class
#####################################################################
class VPRModel(nn.Module):
    def __init__(self, feature_extractor, clustering):
        super(VPRModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.clustering = clustering

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.clustering(x)
        return x
