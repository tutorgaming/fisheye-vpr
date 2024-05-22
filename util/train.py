#!/usr/bin/env python3
"""
Training
"""
# Author : Theppasith N. <tutorgaming@gmail.com>
# Date : 04-Jun-2023
#####################################################################
# Imports
#####################################################################
import os
import numpy as np
from dateutil import tz
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
from typing import Any
from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter


#####################################################################
# Default
#####################################################################
DEFAULT_DICT = {
    "run_name": "default",
    "start_time": "",
    "training_epoch": 20,
    "enable_tensorboard": False,
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
        self.train_date_string = self._today_str()
        self.config["run_name"] = self.run_name

        # Tensorboard Logging
        self.tensorboard_enable = False
        self.tensorboard_writer = None
        self.tensorboard_log_path = self.get_result_path_dir()

        if "enable_tensorboard" in self.config:
            self.tensorboard_enable = self.config["enable_tensorboard"]
            self.tensorboard_writer = SummaryWriter(self.tensorboard_log_path)
            print("Tensorboard Enabled : {}".format(self.tensorboard_enable))
            print("Log Path : {}".format(self.tensorboard_log_path))

    def get_writer(self)->SummaryWriter:
        """
        Return this Training SummaryWriter
        """
        if self.tensorboard_enable:
            return self.tensorboard_writer
        else:
            print("No Tensorboard Enabled - But you request the log - Return None")
            return None

    def _today_str(self)->str:
        """
        Generate Today Folder Name
        For Naming Sort We use YEARMONTHDAY (20221231)
        """
        today = datetime.today()
        today = today.astimezone(tz.gettz('Asia/Bangkok'))
        today_string = today.strftime("%Y%m%d")
        return today_string

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
        epoch_idx,
        batch_idx_count
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

                # Write loss to Tensorboard
                if self.tensorboard_enable is True:
                    self.tensorboard_writer.add_scalar(
                        "Loss/train_batch", loss, batch_idx_count)

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
                batch_idx_count+=1

        # End Training - Calculate Average Loss by divide rounds
        epoch_avg_loss /= len(train_epoch)
        return epoch_avg_loss, batch_idx_count

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
            # print(f"Validating epoch {epoch_idx}")
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
                    # Log
                    tepoch.set_postfix(
                    epoch_idx=epoch_idx,
                    loss=loss.item(),
                    )
                    # Memorize Loss
                    val_losses.append(loss)
            # Calculate loss from list
            avg_val_loss = torch.stack(val_losses).mean().item()

        # print(f"Validating epoch {epoch_idx} - Batch Avg Loss : {avg_val_loss}")
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
        batch_idx_count = 0
        # Training Setting
        target_epoch_count = self.config["training_epoch"]
        # Iterate through every Epochs
        for epoch_idx in tqdm(range(target_epoch_count), desc="Model Training", unit="epoch"):
            # [Train]
            train_dataloader = self.dataset.train_dataloader
            epoch_train_loss, batch_idx_count = self._train_model(
                self.model,
                train_dataloader,
                epoch_idx,
                batch_idx_count
            )
            # [Save]
            self.save_model(self.model, epoch_idx)
            # [Validate]
            val_dataloader = self.dataset.val_dataloader
            epoch_val_loss = self._validate_model(
                self.model,
                val_dataloader,
                epoch_idx
            )
            # [Log]
            if self.tensorboard_enable is True:
                self.tensorboard_writer.add_scalar(
                    "Loss/Train", epoch_train_loss, epoch_idx)
                self.tensorboard_writer.add_scalar(
                    "Loss/Validation", epoch_val_loss, epoch_idx)
            # Append Loss
            epoch_train_loss_list.append(epoch_train_loss)
            epoch_val_loss_list.append(epoch_val_loss)
        # Flush Tensorboard
        if self.tensorboard_enable is True:
            self.tensorboard_writer.flush()

        # Populate Best Model and Output Log
        self.config["epoch_train_loss"] = np.array(epoch_train_loss_list)
        self.config["epoch_val_loss"] = np.array(epoch_val_loss_list)
        # Find the best train model idx
        best_train_loss_idx = epoch_train_loss_list.index(min(epoch_train_loss_list))
        best_val_loss_idx = epoch_val_loss_list.index(min(epoch_val_loss_list))
        self.config["best_train_loss_idx"] = best_train_loss_idx
        self.config["best_val_loss_idx"] = best_val_loss_idx
        self.config["result_path_dir"] = str(self.get_result_path_dir())
        self.config["best_model_train_path"] = Path(os.path.join(
            self.config["result_path_dir"], f"model_{best_train_loss_idx:02d}.pt"
        ))
        self.config["best_model_val_path"] = Path(os.path.join(
            self.config["result_path_dir"], f"model_{best_val_loss_idx:02d}.pt"
        ))
        print("Training Completed !")


    def get_result_path_dir(
        self,
    )->str:
        """
        Generate Result Path Directory
        """
        result_path = f"/workspace/results/{self.train_date_string}/{self.run_name}"
        # Create folder if not exists
        result_pathlib = Path(result_path)
        result_pathlib.mkdir(parents=True, exist_ok=True)
        return result_path

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
        result_path = self.get_result_path_dir()
        # Final Path
        model_file_path = f"{result_path}/{model_file_name}"
        # Save model checkpoint
        torch.save(model.state_dict(), model_file_path)
        # TODO: Find the good way to check the saving completeness
        return model_file_path
