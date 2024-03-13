#!/usr/bin/env python3
"""
Training
"""
# Author : Theppasith N. <tutorgaming@gmail.com>
# Date : 04-Jun-2023
#####################################################################
# Imports
#####################################################################
from dateutil import tz
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from typing import Any
from tqdm import tqdm

#####################################################################
# Default
#####################################################################
DEFAULT_DICT = {
    "run_name": "default",
    "start_time": "",
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
        self.run_name = self._generate_run_name()
        self.config["run_name"] = self.run_name

    def _generate_run_name(
        self
    )->str:
        """
        Generate Run name
        """
        today = datetime.today()
        today = today.astimezone(tz.gettz('Asia/Bangkok'))
        time_string = today.strftime("%d-%b-%Y_%H-%M-%S")
        self.config["start_time"] = time_string
        run_name = "{}{}_NetVLAD{}_{}_{}".format(
            self.model.feature_extractor.name,
            "finetuned" if self.model.feature_extractor.fine_tuning else "",
            self.model.clustering.num_clusters,
            self.dataset.name,
            time_string
        )

        print("Run Name : {}".format(run_name))
        return run_name

    def _check_config(
        self,
        config: dict
    )->bool:
        """
        Check if Configuration is Correct
        """
        REQUIRED_KEY = [
            "run_name", "start_time", "training_epoch"
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
        model,
        train_dataloader,
        epoch_idx
    )->float:
        """
        Training for A SINGLE EPOCH
        return EPOCH Loss
        """
        # Set Model to Training Mode
        model.train()
        # Initiate Epoch avg loss
        epoch_avg_loss = 0.0
        # Initiated Logger
        with tqdm(
            train_dataloader,
            desc="==> Epoch {} Training".format(epoch_idx),
            unit=" batch"
        ) as train_epoch:
            # Iterate through all data batches in train epoch until cover complete dataset
            for batch_idx, batch_data in enumerate(train_epoch):
                # Extract batch data
                train_image_batch, train_label_batch = batch_data

                # Forward Passing
                output_train = self.model(train_image_batch.cuda())
                # Calculate Loss
                loss = self.criterion(output_train.cuda(), train_label_batch.cuda())

                # Memorize Loss for Epoch Loss
                epoch_avg_loss += loss.item()
                # Reset the gradient
                self.optimizer.zero_grad()
                # Calculate Loss - Backpropagation
                loss.backward(retain_graph=True)
                # Optimizer adjust param that's been backproped
                self.optimizer.step()

                # Update TQDM Printer
                train_epoch.set_postfix(
                    epoch_idx=epoch_idx,
                    loss=loss.item(),
                    batch_idx=batch_idx
                )

        # End Training - Calculate Average Loss by divide rounds
        epoch_avg_loss /= len(train_epoch)
        return epoch_avg_loss

    def _validate_model(self,
        model,
        val_dataloader,
        epoch_idx
    ):
        """
        Validation for A SINGLE EPOCH
        return Val EPOCH Loss
        """
        # Model Validation Set
        model.eval()
        # Validation
        val_losses = []
        avg_val_loss = 0.0

        # No Gradient is backproped
        with torch.no_grad():
            print(f"Validating epoch {epoch_idx}")
            with tqdm(
                val_dataloader,
                unit="batch",
                desc="==> Epoch {} Validating".format(epoch_idx)
            ) as tepoch:
                for val_image, val_label in tepoch:
                    # Pass Data point to model
                    output_train = model(val_image.cuda())
                    # Calculate Loss
                    loss = self.criterion(output_train.cuda(), val_label.cuda())
                    # Memorize Loss
                    val_losses.append(loss)
            # Calculate loss from list
            avg_val_loss = torch.stack(val_losses).mean().item()
        return avg_val_loss

    def train(self):
        """
        Train Model and Validate it !
        - Store checkpoint every model
        - TODO : Write the result to SummaryWriter
        """
        # Create Array to Store the States
        epoch_train_loss_list = []
        epoch_val_loss_list = []

        # Training Setting
        target_epoch_count = self.config["training_epoch"]
        # Iterate through every Epochs
        for epoch_idx in tqdm(range(target_epoch_count), desc="Model Training", unit="epoch"):
            # [Train]
            train_dataloader = self.dataset.train_dataloader
            epoch_train_loss = self._train_model(self.model, train_dataloader, epoch_idx)
            # [Save]
            self.save_model(self.model, epoch_idx)
            # [Validate]
            val_dataloader = self.dataset.val_dataloader
            epoch_val_loss = self._validate_model(self.model, val_dataloader, epoch_idx)
            # Append Loss
            epoch_train_loss_list.append(epoch_train_loss)
            epoch_val_loss_list.append(epoch_val_loss)

    def save_model(
        self,
        model,
        epoch_idx:int
    )->Path:
        """
        Save Model with torch.save model state
        """
        # Model Name (Checkpoint name)
        model_file_name = 'model_{:02d}.pt'.format(epoch_idx)
        # Path Setup
        result_path = f"./fisheye-vpr/notebooks/runs/{self.run_name}"
        # Create folder if not exists
        result_pathlib = Path(result_path)
        result_pathlib.mkdir(parents=True, exist_ok=True)
        # Final Path
        model_file_path = f"{result_path}/{model_file_name}"
        # Save model checkpoint
        torch.save(model.state_dict(), model_file_path)
        # TODO: Find the good way to check the saving completeness
        return model_file_path
