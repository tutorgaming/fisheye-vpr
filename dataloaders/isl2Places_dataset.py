#!/usr/bin/env python3
"""
Pytorch Dataset Class for ISL2Places Dataset
Dataset Format is
- classname-x_<float>-y_<float>.jpg
Dataset is Inherited from torchvision Dataset Class
    - ImageFolder Dataset Class
"""
# Author : Theppasith N. <tutorgaming@gmail.com>
# Date : 07-Apr-2023
#####################################################################
# Imports
#####################################################################
import os
import glob
from PIL import Image
from torch.utils.data import Dataset

#####################################################################
# Class
#####################################################################
class ISL2PlacesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(self.root_dir, '*.jpg'))
        self.image_paths.sort()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
