#!/usr/bin/env python3
"""
Entry Point for Traning
Visual Place Recognition Pipeline
"""
# Author : Theppasith N. <tutorgaming@gmail.com>
# Date : 6-Jan-2024
#####################################################################
# Imports
#####################################################################
from dataloaders.isl2_3places_fisheye import ISL2_3Places_Fisheye_Dataset


class DatasetTester(object):
    def __init__(self):
        self.dataset = ISL2_3Places_Fisheye_Dataset()

    def test(self):
        # Test Dataset Initialization
        assert(self.dataset.train_dataset is not None)
        # Test Dataloader Initialization
        assert(self.dataset.train_dataloader is not None)


if __name__ == "__main__":
    tester = DatasetTester()
    tester.test()
    print("Dataset Test Passed !")


