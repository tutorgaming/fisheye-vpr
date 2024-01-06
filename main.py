#!/usr/bin/env python3
"""
Entry Point for Traning
Visual Place Recognition Pipeline
"""
# Author : Theppasith N. <tutorgaming@gmail.com>
# Date : 4-Jun-2023
#####################################################################
# Imports
#####################################################################
import torch
from trainer import Trainer
from evaluator import Evaluator

# Main Model
from models.vpr_model import VPRModel

# Clustering Model
from models.clustering.netvlad import NetVLAD

# Features Extractor
from models.encoders.rps      import RPSEncoder
from models.encoders.vgg16    import VGG16Encoder
from models.encoders.resnet18 import Resnet18Encoder

# Dataset & Dataloader Classes
from dataloaders.paris6k import Paris6K_Dataset

#####################################################################
# Class
#####################################################################

class VPR(object):
    """
    Main Class for Conducting Training and Evaluating
    Visual Place Recognition Pipeline
    """
    def __init__(self, config):
        # Parse Configuration
        data_config, feature_config, clustering_config, model_config, train_config, eval_config = self.parse_config(config)
        # Select Dataset
        self.feature_extractor = Resnet18Encoder()
        self.clustering = NetVLAD()
        self.model   = VPRModel()
        self.dataset = Paris6K_Dataset()

    def parse_config(self, config):
        """
        Parse Configuration
        """
        data_config       = config["data"]
        feature_config    = config["feature"]
        clustering_config = config["clustering"]
        model_config      = config["model"]
        train_config      = config["train"]
        eval_config       = config["eval"]

        return data_config, feature_config, clustering_config, model_config, train_config, eval_config


if __name__ == "__main__":
    # Create VPS Class
    config = {
        "Hello" : "World",
    }
    VisualPlaceRecognition = VPR(config)

    # Start things
    VisualPlaceRecognition.train()
    VisualPlaceRecognition.evaluate()
