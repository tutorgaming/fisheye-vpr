#!/usr/bin/env python3
"""
Paris6K Dataset and Dataloaders
"""
# Author : Theppasith N. <tutorgaming@gmail.com>
# Date : 04-Jun-2023
#####################################################################
# Imports
#####################################################################
import torch
import numpy as np
import torchvision
from sklearn.model_selection import train_test_split

#####################################################################
# Class
#####################################################################
class Paris6K_Dataset(object):
    def __init__(self):
        # Dataset Path
        self.path = "/workspace/Datasets/paris6k"

        # Train Test Split Ratio
        self.train_ratio = 0.80
        self.val_ratio   = 0.10
        self.test_ratio  = 0.10
        assert(self.train_ratio + self.val_ratio + self.test_ratio == 1.0)

        # Dataset
        self.dataset_transform = self.create_transform()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Dataloader
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None

        # Initialization
        self.initialize()

    def initialize(self):
        """
        Initialize Dataset and DataLoaders
        """
        self.create_dataset()
        self.create_dataloader()

    def create_transform(self):
        """
        Transform Image !
        """
        torchvision.transforms.Compose([
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.ToTensor()
        ])

    def create_dataset(self):
        """
        Make Dataset Object, Create Train-Val-Test Split of Dataset
        """
        # Load ImageFolder Dataset
        paris6k_dataset = torchvision.datasets.ImageFolder(
            root=self.path,
            transform=self.dataset_transform
        )
        # Split TRAINVAL/TEST
        paris6k_trainval_idx, paris6k_test_idx = train_test_split(
            np.arange(len(paris6k_dataset)),
            test_size=self.test_ratio,
            shuffle=True,
            stratify=paris6k_dataset.targets
        )
        # Split Train Val
        paris6k_train_idx, paris6k_val_idx = train_test_split(
            paris6k_trainval_idx,
            test_size=(self.val_ratio/self.train_ratio),
            shuffle=True,
            stratify=np.array(paris6k_dataset.targets)[paris6k_trainval_idx]
        )

        self.train_dataset = torch.utils.data.Subset(
            paris6k_dataset,
            paris6k_train_idx
        )

        self.val_dataset = torch.utils.data.Subset(
            paris6k_dataset,
            paris6k_val_idx
        )

        self.test_dataset = torch.utils.data.Subset(
            paris6k_dataset,
            paris6k_test_idx
        )

    def create_dataloader(self):
        """
        Create Dataloader
        """
        # Dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=8,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )

        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=100,
            shuffle=False,
            num_workers=0
        )
