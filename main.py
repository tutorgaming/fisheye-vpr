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
import logging
import torch
from datetime import datetime
from dateutil import tz
from pathlib import Path
import yaml

# Custom Modules
from util.train import Trainer
from evaluator import Evaluator

# Main Model
from models.vpr_model import VPRModel

# Clustering Model
from models.clustering.netvlad import NetVLAD
from models.clustering.hloc_netvlad import HLOCNetVLAD

# Features Extractor
# from models.encoders.rps      import RPSEncoder
from models.encoders.vgg16    import VGG16Encoder
from models.encoders.resnet18 import Resnet18Encoder
from models.encoders.hloc_vgg16 import HLOCVGG16Encoder

# Dataset & Dataloader Classes
from dataloaders.paris6k import Paris6K_Dataset
from dataloaders.triplet_dataset import TripletImageDataset
from dataloaders.isaac_office_all_fisheye import IsaacOffice_All_Fisheye_Dataset
from dataloaders.eng3_floor1_fisheye import ENG3_Floor1_Fisheye_Dataset
from dataloaders.isl2_3places_fisheye import ISL2_3Places_Fisheye_Dataset
from dataloaders.isl2_3places_pinhole import ISL2_3Places_Pinhole_Dataset
from dataloaders.isl2_places_fisheye import ISL2_Places_Fisheye_Dataset
from dataloaders.all_fisheye import All_Fisheye_Dataset

# Loss Function / Criterion
from models.loss_function.HardTripletLoss import HardTripletLoss

#####################################################################
# Dict
#####################################################################
EXTRACTOR_DICT = {
    # "rps"      : RPSEncoder,
    "vgg16"    : VGG16Encoder,
    "resnet18" : Resnet18Encoder,
    "hloc_vgg16": HLOCVGG16Encoder,
}
DATASET_DICT = {
    "paris6k" : Paris6K_Dataset,
    "triplet" : TripletImageDataset,
    "eng3_floor1_fisheye" : ENG3_Floor1_Fisheye_Dataset,
    "isaac_office_all_fisheye" : IsaacOffice_All_Fisheye_Dataset,
    "isl2_3places_fisheye" : ISL2_3Places_Fisheye_Dataset,
    "isl2_3places_pinhole" : ISL2_3Places_Pinhole_Dataset,
    "isl2_places_fisheye" : ISL2_Places_Fisheye_Dataset,
    "all_fisheye" : All_Fisheye_Dataset,
}

CLUSTERING_DICT = {
    "netvlad" : NetVLAD,
    "hloc_netvlad" : HLOCNetVLAD,
}

CRITERIAN_DICT = {
    "triplet" : torch.nn.TripletMarginLoss,
    "hardtripletloss" : HardTripletLoss,
}

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
        self.config_dict = config
        # Select Dataset
        self.feature_extractor = None
        self.clustering = None
        self.dataset = None
        self.model   = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

        # Do the Setup
        self.setup(self.config_dict)

    def setup(self, config):
        """Setup the Dataset, Model, Training and Evaluation Pipeline

        Args:
            config (dict): setup dict
        """
        # Select Dataset
        dataset_key = config["dataset"]
        self.dataset = DATASET_DICT[dataset_key]()

        # Feature Extractor
        feature_extractor_key = config["model_config"]["feature_extractor"]["model"]
        feature_extractor_config = config["model_config"]["feature_extractor"]
        self.feature_extractor = EXTRACTOR_DICT[feature_extractor_key](
            fine_tuning=feature_extractor_config["fine_tuning"],
        )

        # Clustering
        clustering_key = config["model_config"]["clustering"]["model"]
        clustering_config = config["model_config"]["clustering"]
        if clustering_key == "hloc_netvlad":
            self.clustering = HLOCNetVLAD(
                num_clusters=clustering_config["num_clusters"],
                desc_dim=clustering_config["desc_dim"],
                score_bias=None,#clustering_config["score_bias"],
                intranorm=None,#clustering_config["intranorm"],
                whiten=None, #clustering_config["whiten"],
            )
        else:
            self.clustering = NetVLAD(
                num_clusters=clustering_config["num_clusters"],
                desc_dim=clustering_config["desc_dim"],
                alpha=clustering_config["alpha"],
                normalize_input=clustering_config["normalize_input"],
            )


        # Model
        self.model = VPRModel(
            feature_extractor=self.feature_extractor,
            clustering=self.clustering,
        )

        # Loss Function
        loss_key = config["loss"]["name"]
        self.criterion = CRITERIAN_DICT[loss_key](
            margin=config["loss"]["margin"],
            hardest=config["loss"]["hardest"],
            squared=config["loss"]["squared"],
        )

        # Training Optimizer
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.0001,
            weight_decay=0.001,
            momentum=0.9
        )

        # Training Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=5,
            gamma=0.5
        )

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(Path(config_path)) as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # Iterate to the config file for the sessions
    configs_path = Path("/workspace/fisheye-vpr/config")

    # Glob the Yaml File
    # For each yaml file do the entire training
    for config_file in configs_path.glob("*.yaml"):
        print("\n"*10)

        # Load Configuration
        config = load_config(config_file)

        print(f"Training with Config : {config_file}")

        # Create Model
        VPR_MODEL = VPR(config)

        # Create Trainer
        trainer = Trainer(
            dataset=VPR_MODEL.dataset,
            model=VPR_MODEL.model.to("cuda"),
            criterion=VPR_MODEL.criterion,
            optimizer=VPR_MODEL.optimizer,
            scheduler=VPR_MODEL.scheduler,
            writer=None,
            config=VPR_MODEL.config_dict
        )
        complete_log_dict = trainer.train()

        # Print the space for the next training
        # Pytorch release GPU
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Release the memory (RAM , GPU RAM)
        del VPR_MODEL
        del trainer
