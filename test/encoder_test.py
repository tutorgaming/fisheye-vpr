#!/usr/bin/env python3
"""
Class to Check that the encoder model is working perfectly
to extract local feature from the model
"""
#####################################################################
# Imports
#####################################################################
import numpy as np
from tqdm import tqdm
from dataloaders.isl2_3places_fisheye import ISL2_3Places_Fisheye_Dataset
from models.encoders.resnet18 import Resnet18Encoder
from models.encoders.vgg16 import VGG16Encoder


class EncoderTester(object):
    def __init__(self):
        self.encoder_pool = [Resnet18Encoder, VGG16Encoder]

        # Load Dataset and DataLoaders
        self.dataset = ISL2_3Places_Fisheye_Dataset()

        # Test Forwarding
        self.test_encoder()

    def test_encoder(self):
        """
        Given the encoder model, pass the dataset through the encoder
        """
        for encoder in self.encoder_pool:
            selected_encoder = encoder()

            with tqdm(self.dataset.test_dataloader, unit=" batch") as tepoch:
                # Divide all data into many batches
                # and feed batch by batch until all data covered
                for train_image, train_label in tepoch:
                    print("Accessing Picture : {}".format(train_label))
                    print("Accessing Picture : {}".format(np.shape(train_image)))
                    encoder_output = selected_encoder(train_image)
                    print("Encoder Output : {}".format(np.shape(encoder_output)))

if __name__ == "__main__":
    ENCODERTEST = EncoderTester()