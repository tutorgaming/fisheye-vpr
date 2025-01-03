#!/usr/bin/env python3
"""
Pytorch Implementation of NetVLAD Part
"""
# Author : Theppasith N. <tutorgaming@gmail.com>
# Date : 14-Apr-2023
#####################################################################
# Imports
#####################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

#####################################################################
# Class
#####################################################################
class NetVLAD(nn.Module):
    """
    NetVLAD Implementation
    """
    def __init__(self,
            num_clusters=6,
            desc_dim=128,
            alpha=100.0,
            normalize_input=True,):

        # Initialize the Module
        super(NetVLAD, self).__init__()
        self.name = "NetVLAD"
        # Parameters
        self.num_clusters = num_clusters
        self.dim = desc_dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.DEBUG = False

        # Layers
        # Latent Convolution (add weight and bias to every latent)
        self.conv = nn.Conv2d(
            self.dim,
            self.num_clusters,
            kernel_size=(1, 1),
            bias=True
        )

        # Where are those Centroids ?
        self.centroids = nn.Parameter(
            torch.rand(self.num_clusters, self.dim)
        )

        # Initialize Param
        self._init_params()

        # Prompt
        self.show_config()

    def show_config(self):
        """
        Show the Module settings
        """
        print("===========================================")
        print("NetVLAD Module initialized !")
        print("===========================================")
        print(" - self.num_clusters : {}".format(self.num_clusters))
        print(" - self.dim : {}".format(self.dim))
        print(" - self.alpha : {}".format(self.alpha))
        print(" - self.normalize_input : {}".format(self.normalize_input))
        print("===========================================")

    def _init_params(self):
        """
        Initialize Parameters for Conv and Centroid
        """
        # Convolution Initialize Weight
        # Unsqueeze for Extend the dimension from ( K cluster, D-feature dimension ) = (K x D)
        # to (K x D x 1 x 1)
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        # Convolution Initialize Bias
        self.conv.bias = nn.Parameter(
            -self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        """
        Forward Passing for NetVLAD Layer
        input x -> Local Feature Description
        W*H*D map interpreted as NxD Local Descriptor "x"
        """
        # Store the Shape of first 2 dim (batch size, channels)
        BATCH_SIZE, CHANNEL = x.shape[0:2]
        # Decide to Normalize or not ?
        if self.normalize_input:
            # Normalize Across Descriptor Dim (across vector's components)
            x = F.normalize(x, p=2, dim=1)

        ################################################
        # Soft Assignment (A)
        ################################################
        soft_assignment = self.conv(x) # WkXi + Bk (output_shape = (cluster,W,H))
        # Readjust the shape to (N rows x K cluster x any)
        soft_assignment = soft_assignment.view(BATCH_SIZE, self.num_clusters, -1) # (Cluster x w*h)
        # Create result from group assignment -> a(x) (dim 1 mean in the row vec)
        soft_assignment = F.softmax(soft_assignment, dim=1)

        ################################################
        # Create VLAD Core
        ################################################
        # Create Flatten X (X which is directly comes from feature encoder)
        x_flatten = x.view(BATCH_SIZE, CHANNEL, -1) # (N x FeatureDim x (w*h))

        # Calculate Residual
        x_flatten_adjusted = x_flatten.expand(self.num_clusters, -1, -1, -1)
        x_flatten_adjusted = x_flatten_adjusted.permute(1,0,2,3)
        # Now X_Flatten_Adjusted Shape = ( torch.Size([1, 6, 128, 16]) )
        # = torch.Size([1, CLUSTER_SIZE , FEATURE_SIZE , (WxH) ])
        centroid_adjusted = self.centroids.expand(x_flatten.size(-1), -1, -1) # (w*h,-1,-1)
        centroid_adjusted = centroid_adjusted.permute(1,2,0)
        centroid_adjusted = centroid_adjusted.unsqueeze(0)
        residual = x_flatten_adjusted - centroid_adjusted

        # Calculate Summation
        vlad = residual * soft_assignment.unsqueeze(2)
        vlad = vlad.sum(dim=-1)
        # Intra-Normalization
        vlad = F.normalize(vlad, p=2, dim=2)
        vlad = vlad.view(x.size(0), -1)  # flatten

        # L2 normalize
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
