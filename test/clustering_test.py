#!/usr/bin/env python3
import torch
import numpy as np
from tqdm import tqdm
from dataloaders.isl2_3places_fisheye import ISL2_3Places_Fisheye_Dataset
from models.clustering.netvlad import NetVLAD
from models.encoders.resnet18 import Resnet18Encoder
from models.vpr_model import VPRModel
from models.loss_function.HardTripletLoss import HardTripletLoss


class ClusteringTest(object):
    def __init__(self):
        # Data to be feed
        self.dataset = ISL2_3Places_Fisheye_Dataset()
        # Model to be trained
        self.encoder = Resnet18Encoder()
        netvlad_config = {
            "num_clusters": 8,
            "desc_dim": 512, # Up to the Feature Extraction Module
            "alpha": 100.0,
            "normalize_input": True,
        }
        self.clustering = NetVLAD(**netvlad_config)
        self.model = VPRModel(self.encoder, self.clustering)
        # Criterion (Loss Function) to be used
        self.criterion = HardTripletLoss()
        # Optimizer to be used
        self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=0.0001,
                weight_decay=0.001,
                momentum=0.9
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
        # Start the process
        self.test_clustering()

    def test_clustering(self):

        epoch_loss = 0.0
        batch_idx = 0.0
        with tqdm(self.dataset.train_dataloader, unit=" batch") as tepoch:
            # Divide all data into many batches
            # and feed batch by batch until all data covered
            for train_image, train_label in tepoch:
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
                tepoch.set_postfix(epoch=tepoch, loss=loss.item(), batch_idx=batch_idx)
                batch_idx += 1
            # Calculate Loss Per Epoch
            epoch_loss /= len(tepoch)



if __name__ == "__main__":
    CLUSTERING_TEST = ClusteringTest()
