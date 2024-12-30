#!/usr/bin/env python3
"""
Triplet Dataset
- torchvision ImageFolder with Triplet Data
"""
# Author : Tutorgaming <Tutorgaming@gmail.com>
# Date : 2-Sept-2024
#####################################################################
# Imports
#####################################################################
import logging
from typing import Any, Tuple
from torchvision.datasets import ImageFolder

import numpy as np
from pathlib import Path

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S'
)

#####################################################################
# Class Definition
#####################################################################
class TripletImageDataset(ImageFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

        root/fish/x42x4x.png
        root/fish/x1xy.png
        root/fish/[...]/xx1414z.png

    and will return the triplet of (query, positive, negative) samples

    This class inherits from :class:`~torchvision.datasets.ImageFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory path.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
            An error is raised on empty folders if False (default).
        allow_itself (bool): Allow selecting itself as positive index
        precalculated (bool): Pre-calculate the triplets and store them in memory in the initialization
            for faster access

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
        np_targets (np.array): Numpy Array of Targets
        np_samples (np.array): Numpy Array of Samples
    """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        allow_itself=False,
        precalculated=False
    ):
        # Initialize Parent Class
        super(TripletImageDataset, self).__init__(root, transform, target_transform)
        # Additional Attributes
        self.np_targets = np.array(self.targets)
        self.np_samples = np.array(self.samples)
        self.allow_itself = allow_itself
        self.precalculated = precalculated

        # Precalculated
        if self.precalculated:
            self.triplets = []
            self._pre_calculate_triplets()

    def _pre_calculate_triplets(self):
        """
        Pre-Calculate Triplets for Faster Access on initialization if specified
        """
        for i in range(len(self)):
            query_path, query_target = self.samples[i]
            pos_path, pos_target = self.samples[self.get_positive_idx((query_path, i, query_target))]
            neg_path, neg_target = self.samples[self.get_negative_idx((query_path, i, query_target))]
            query_tuple = (query_path, query_target)
            positive_tuple = (pos_path, pos_target)
            negative_tuple = (neg_path, neg_target)
            self.triplets.append((query_tuple, positive_tuple, negative_tuple))

    def get_positive_idx(self, query):
        """
        Given the Query (Anchor)
        randomly select the positive index in the same class as query
        Args:
            query (Dataset Item): Data Extract from Dataset in format (data, class_idx)
            allow_itself (bool, optional): allow selecting itself. Defaults to False.
        """
        _, index, class_idx = query
        positive_db_idx = np.where(self.np_targets == class_idx)[0]

        # Randomly select the positive index
        positive_idx = None
        pick_count = 0
        while True:
            positive_idx = np.random.choice(positive_db_idx)
            if positive_idx == index:   # Pick itself
                # Allow itself when randomly picked itself
                if self.allow_itself:        # Allow itself
                    break
            else:                       # Pick other
                break
            if pick_count > 100:
                raise RuntimeError("Cannot find positive index in 100 tries - go buy a lottery ticket")
            pick_count += 1

        return positive_idx

    def get_negative_idx(self, query):
        """
        Given the Query (Anchor)
        randomly select the negative index
        from any class other than the query class

        Args:
            query (_type_): _description_
        """
        _, index, class_idx = query
        negative_db_idx = np.where(self.np_targets != class_idx)[0]
        negative_idx = np.random.choice(negative_db_idx)
        if negative_idx == index:
            raise RuntimeError("Negative Index is the same as Query Index")
        return negative_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index of data

        Returns:
            3-tuple: (query, positive, negative) samples
            - query: Desire Sample (Anchor Query)
            - positive: Positive Sample (Same Class with query)
            - negative: Negative Sample (Different Class with query)
        """
        if not self.precalculated:
            query_path, query_target = self.samples[index]
            pos_path, pos_target = self.samples[self.get_positive_idx((query_path, index, query_target))]
            neg_path, neg_target = self.samples[self.get_negative_idx((query_path, index, query_target))]
        else:
            (query_path, query_target), (pos_path, pos_target), (neg_path, neg_target) = self.triplets[index]

        query_sample = self.loader(query_path)
        positive_sample = self.loader(pos_path)
        negative_sample = self.loader(neg_path)

        # Transform Data
        if self.transform is not None:
            query_sample = self.transform(query_sample)
            positive_sample = self.transform(positive_sample)
            negative_sample = self.transform(negative_sample)
        # Transform Label (Target)
        if self.target_transform is not None:
            query_target = self.target_transform(query_target)
            pos_target = self.target_transform(pos_target)
            neg_target = self.target_transform(neg_target)

        # Return Tuple
        query_tuple = (query_sample, query_target)
        positive_tuple = (positive_sample, pos_target)
        negative_tuple = (negative_sample, neg_target)

        return query_tuple, positive_tuple, negative_tuple
