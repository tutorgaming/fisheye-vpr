#!/usr/bin/env python3

import numpy as np
from tqdm import tqdm
from dataloaders.isl2_3places_fisheye import ISL2_3Places_Fisheye_Dataset
from models.clustering.netvlad import NetVLAD
from models.encoders.resnet18 import Resnet18Encoder
from models.vpr_model import VPRModel


class ClusteringTest(object):
    def __init__(self):
        # Data to be feed
        self.dataset = ISL2_3Places_Fisheye_Dataset()
        # Model to be trained
        self.encoder = Resnet18Encoder()
        self.clustering = NetVLAD()
        self.model = VPRModel(self.encoder, self.clustering)
        # Criterion (Loss Function) to be used
        self.criterion = None
        # Optimizer to be used
        self.optimizer = None
        # Start the process
        test_clustering()

    def test_clustering(self):
        with tqdm(self.dataset.test_dataloader, unit=" batch") as tepoch:
            # Divide all data into many batches
            # and feed batch by batch until all data covered
            for train_image, train_label in tepoch:
                print("Accessing Picture : {}".format(train_label))
                print("Accessing Picture : {}".format(np.shape(train_image)))
                output_train = self.model(train_image)
                print("Model Output : {}".format(np.shape(output_train)))
                    # Calculate Loss
                loss = self.criterion(output_train.cuda(), train_label.cuda())
                # Memorize Loss for Epoch Loss
                epoch_loss += loss.item()
                # Reset the gradient
                self.optimizer.zero_grad()
                # Calculate Loss - Backpropagation
                loss.backward(retain_graph=True)
                # Optimizer adjust param that's been backproped
                self.optimizer.step()
                tepoch.set_postfix(epoch=epoch, loss=loss.item(), batch_idx=batch_idx)
                batch_idx += 1
            # Calculate Loss Per Epoch
            epoch_loss /= len(tepoch)


if __name__ == "__main__":
    CLUSTERING_TEST = ClusteringTest()
