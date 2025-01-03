#!/usr/bin/env python3
"""
Eng3 Floor1 Fisheye Dataset and Dataloaders
"""
# Author : Theppasith N. <tutorgaming@gmail.com>
# Date : 14-Mar-2024
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
class All_Fisheye_Dataset(object):
    def __init__(self):
        self.name = "All_Fisheye"
        # Dataset Path
        self.path = "/workspace/Datasets/all_fisheye_ever"
        # Train Test Split Ratio
        self.train_ratio = 0.60
        self.val_ratio   = 0.20
        self.test_ratio  = 0.20
        assert(self.train_ratio + self.val_ratio + self.test_ratio == 1.0)

        # Dataset
        self.dataset_transform = self.create_transform()
        self.dataset = None
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
        # Crop image to have center square
        crop = lambda img: torchvision.transforms.functional.crop(
            img,
            top=0,
            left=260,
            height=700,
            width=700,
        )

        # Resize square image
        image_size = 640
        resize = torchvision.transforms.Resize((image_size,image_size))

        # Random Rotation
        random_rotate = torchvision.transforms.RandomRotation(degrees=90)

        return torchvision.transforms.Compose([
            crop,
            resize,
            random_rotate,
            torchvision.transforms.ToTensor()
        ])

    def create_dataset(self):
        """
        Make Dataset Object, Create Train-Val-Test Split of Dataset
        """
        # Load ImageFolder Dataset
        self.dataset = torchvision.datasets.ImageFolder(
            root=self.path,
            transform=self.dataset_transform
        )
        # Split TRAINVAL/TEST
        train_val_idx, test_idx = train_test_split(
            np.arange(len(self.dataset)),
            test_size=self.test_ratio,
            shuffle=True,
            stratify=self.dataset.targets
        )
        # Split Train Val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=(self.val_ratio/self.train_ratio),
            shuffle=True,
            stratify=np.array(self.dataset.targets)[train_val_idx]
        )

        self.train_dataset = torch.utils.data.Subset(
            self.dataset,
            train_idx
        )

        self.val_dataset = torch.utils.data.Subset(
            self.dataset,
            val_idx
        )

        self.test_dataset = torch.utils.data.Subset(
            self.dataset,
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
            drop_last=False
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

    def get_same_class_data(self, class_label):
        """
        Return the Subset of Dataset of the Same Class Label
        """
        target_indices = [idx for idx, (img,label) in enumerate(self.dataset) if label == class_label]
        target_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(self.dataset, target_indices),
            batch_size=8,
            shuffle=False,
            num_workers=0
        )
        return target_loader