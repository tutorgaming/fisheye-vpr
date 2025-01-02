#!/usr/bin/env python3
"""
Isaac Office Fisheye Dataset for all Lighting Condition
"""
# Author : Theppasith N. <tutorgaming@gmail.com>
# Date : 09-Jun-2023
#####################################################################
# Imports
#####################################################################
import torch
import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
from dataloaders.triplet_dataset import TripletImageDataset
######################################################################
# Class
######################################################################
class IsaacOffice_All_Fisheye_Dataset(object):
    """
    Dataset of All Fisheye Images with all Lighting condition
    """
    def __init__(self):
        self.name = "IsaacOfficeAll_Fisheye"
        # Dataset Path
        self.path = "/workspace/Datasets/isaac_office_dataset_rev1/FisheyeCamera"

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
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize((256,256)),
            torchvision.transforms.ToTensor()
        ])

    def create_dataset(self):
        """
        Make Dataset Object, Create Train-Val-Test Split of Dataset
        """
        # Load ImageFolder Dataset
        dataset = TripletImageDataset(
            root=self.path,
            transform=self.dataset_transform
        )
        # Split TRAINVAL/TEST
        train_val_idx, test_idx = train_test_split(
            np.arange(len(dataset)),
            test_size=self.test_ratio,
            shuffle=True,
            stratify=dataset.targets
        )
        # Split Train Val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=(self.val_ratio/self.train_ratio),
            shuffle=True,
            stratify=np.array(dataset.targets)[train_val_idx]
        )

        self.train_dataset = torch.utils.data.Subset(
            dataset,
            train_idx
        )

        self.val_dataset = torch.utils.data.Subset(
            dataset,
            val_idx
        )

        self.test_dataset = torch.utils.data.Subset(
            dataset,
            test_idx
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
            batch_size=8,
            shuffle=False,
            num_workers=0,
            drop_last=True
        )

        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0
        )

######################################################################
# Main (Test)
######################################################################
if __name__ == "__main__":
    DSET = IsaacOffice_All_Fisheye_Dataset()