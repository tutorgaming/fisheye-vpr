#!/usr/bin/env python3
"""
HLOC VGG-16 Encoder Class
- To Extract Local Feature Patch from Images
"""
# Author : Theppasith N. <tutorgaming@gmail.com>
# Date : 22-Jun-2024
#####################################################################
# Imports
#####################################################################
import os
import torch
import torch.nn as nn
from torchvision.models import vgg16
from pathlib import Path

# Matlab Layer weights
from scipy.io import loadmat

#####################################################################
# Class
#####################################################################
class HLOCVGG16Encoder(nn.Module):
    def __init__(self, fine_tuning=False):
        super(HLOCVGG16Encoder, self).__init__()
        # Config
        self.name = "vgg16"
        self.fine_tuning = fine_tuning
        print("[HLOC-VGG16] Loading Pretrained Model")
        # Load Pretrained Model
        encoder = list(vgg16().children())[0]

        # Remove last ReLU + MaxPool2d
        self.feature_extractor = nn.Sequential(*list(encoder.children())[:-2])

        # Load MATLAB Weights
        checkpoint_path = "/workspace/fisheye-vpr/models/weights/Pitts30K_struct.mat"
        print("[HLOC-VGG16] MATLAB Weights : ", checkpoint_path)
        # Check if Path is exist
        if not Path("/workspace/fisheye-vpr/models/weights/Pitts30K_struct.mat").exists():
            print("[HLOC-VGG16] Please Download the MATLAB Weights from HLOC Repo")
        mat = loadmat(
            checkpoint_path,
            struct_as_record=False,
            squeeze_me=True
        )
        self.assign_matlab_weights(mat)

        # Assemble model
        # Freeze Weight or Training more ?
        if self.fine_tuning:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        last_dim_size = list(self.feature_extractor.parameters())[-1].shape[0]
        print("[HLOC-VGG16] Output Dim Size: {}".format(last_dim_size))

    def forward(self, x):
        return self.feature_extractor(x)

    def assign_matlab_weights(self, mat_weight):
        """
        Assign MATLAB weights to the corresponding layers
        """
        # Match the Weight and Corresponding Layer
        pairs = zip(
            self.feature_extractor.children(),
            mat_weight["net"].layers
        )
        # Iterate and Assign over layers
        # Magically, I already checked that the conv layer is adjusted already
        # If we consider only conv
        for idx, (layer, mat_layer) in enumerate(pairs):
            if isinstance(layer, nn.Conv2d):
                imported_layer_weight = mat_layer.weights[0] # S x S x IN x OUT
                imported_layer_bias = mat_layer.weights[1] # OUT

                # Prepare for PyTorch
                imported_layer_weight = torch.tensor(
                    imported_layer_weight
                ).float().permute([3, 2, 0, 1])

                imported_layer_bias = torch.tensor(
                    imported_layer_bias
                ).float()

                layer.weight = nn.Parameter(imported_layer_weight)
                layer.bias = nn.Parameter(imported_layer_bias)
                print("[HLOCVGG16] CONV2D in Layer {} Weights and Bias Imported".format(idx))
