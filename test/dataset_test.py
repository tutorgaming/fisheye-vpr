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
from tqdm import tqdm
import numpy as np

class DatasetTester(object):
    def __init__(self):
        self.dataset = ISL2_3Places_Fisheye_Dataset()

    def test(self):
        # Test Dataset Initialization
        assert(self.dataset.train_dataset is not None)
        # Test Dataloader Initialization
        assert(self.dataset.train_dataloader is not None)

        print("Batch Size = ", self.dataset.train_dataloader.batch_size)

        train_data_length = len(self.dataset.train_dataset)
        print("train_data_length", train_data_length)
        train_dataloader_length = len(self.dataset.train_dataloader)
        print("train_dataloader_length", train_dataloader_length)

        sample_data, sample_label = self.dataset.train_dataset[0]
        print("Sample Label Class : {}".format(sample_label))
        print("Sample Data : {}".format(np.shape(sample_data)))

        with tqdm(self.dataset.test_dataloader, unit=" batch") as tepoch:
            # Divide all data into many batches
            # and feed batch by batch until all data covered
            for train_image, train_label in tepoch:
                print("Accessing Label   : {}".format(np.shape(train_label)))
                print("Accessing Picture : {}".format(np.shape(train_image)))

if __name__ == "__main__":
    tester = DatasetTester()
    tester.test()
    print("Dataset Test Passed !")


