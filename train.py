#!/usr/bin/env python3
"""
Training
"""
# Author : Theppasith N. <tutorgaming@gmail.com>
# Date : 04-Jun-2023
#####################################################################
# Imports
#####################################################################
import torch
import torch.nn as nn
from typing import Any
from tqdm import tqdm

#####################################################################
# Default
#####################################################################
DEFAULT_DICT = {
    "run_name": "default",
    "start_time": None,
    "netvlad_config": {

    }
    "training_epoch": 20
}

#####################################################################
# Class
#####################################################################
class Trainer(object):
    def __init__(
        self,
        dataset: Any,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any=None,
        writer: Any=None,
        config: dict=DEFAULT_DICT
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.writer = writer
        self.dataset = dataset

    def _check_config(
        self,
        config: dict
    )->bool:
        """
        Check if Configuration is Correct
        """
        REQUIRED_KEY = [
            "run_name", "start_time", "netvlad_config", "training_epoch"
        ]
        # Access Config Keys
        config_keys = list(config.keys())
        for req_key in REQUIRED_KEY:
            if req_key in config_keys:
                continue
            else:
                return False
        return True

    def _train_model(
        self,
        train_dataloader
    )->float:
        """
        Training for A SINGLE EPOCH
        return EPOCH Loss
        """
        loss = 0.0
        #
        # Iterate through all data batches until cover complete dataset

        return loss

    def _validate_model(self, val_dataloader):
        pass

    def train(self):
        # Create Array to Store the States
        epoch_train_loss_list = []
        epoch_val_loss_list = []
        # Path Setup
        result_path = f"./runs/{self.run_name}"
        # Training Setting
        target_epoch_count = self.config["training_epoch"]
        # Iterate through every Epochs
        for epoch in tqdm(range(target_epoch_count), desc="Model Training", unit="epoch"):
            # Train model with many databatches
            epoch_train_loss = self.train_model()
            # Save Model every epoch
            model_file_path = f"{result_path}/"
            torch.save(model.stat_dict(), model_file_path)
            # Append Losses
            epoch_train_loss_list.append(epoch_train_loss)

        self.validate_model()