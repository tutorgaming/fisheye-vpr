#!/usr/bin/env python3
import torch
from pathlib import Path

class Evaluator(object):
    """
    Perform the Visual Place Recognition Evaluation
    """
    def __init__(self, model_path, dataset):
        # Model
        self.model_path = model_path
        self.model = None

        # Dataset
        self.dataset = dataset
        self.database = []
        self.database_labels = []

        # Setup
        self.setup()

    def setup(self):
        """Setup the Evaluation Pipeline
        """
        # Load the Model
        self.load_model(self.model_path)
        # Load the Dataset
        self.prepare_database(self.dataset)

    def prepare_database(self, dataset_class):
        """Extract the global descriptor for all images in the database"""
        train_dataloader = dataset_class.train_dataloader
        temp_database = []
        temp_labels = []

        for train_batch in train_dataloader:
            descs, labels = self.generate_global_descriptors(train_batch)
            temp_database.append(descs)
            temp_labels.append(labels)

        self.database = torch.cat(temp_database, dim=0)
        self.database_labels = torch.cat(temp_labels, dim=0)

    def load_model(self, path):
        """Load the Model from the path
        """
        # Load the Model from the Model Path
        self.model = torch.load(path)

        # Cast to GPU if Available
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # Cast to evaluation mode (no gradient)
        self.model.eval()

        print("[Evaluator] Model Loaded and set to EvalMode Successfully")

    def generate_global_descriptors(self, batch):
        """Generate Global Descriptors for the Batch

        Args:
            batch (tuple): Batch Data (images, labels)

        Returns:
            torch.Tensor: Global Descriptors
        """
        # Unpack batch data
        images, labels = batch

        # Move images to GPU if available
        if torch.cuda.is_available():
            images = images.cuda()

        # Extract descriptors and return
        desc_batch = self.model(images)

        return desc_batch, labels

    def evaluate(self, k=10):
        """
        Evaluate the Model on test dataset

        Args:
            k (int): Number of top matches to consider

        Returns:
            float: Average recall@k score
        """
        # Initialize metrics
        total_recall = 0.0
        total_samples = 0
        results = []

        # Iterate through test dataloader
        test_dataloader = self.dataset.test_dataloader
        print(f"[Evaluator] Starting evaluation with k={k}")

        for batch_idx, test_batch in enumerate(test_dataloader):
            # Generate descriptors for test batch
            test_desc_batch = self.generate_global_descriptors(test_batch)

            # Perform Recall at K Matching
            batch_recall = self.recall_at_k(test_desc_batch, self.database, k=k)

            # Update metrics
            batch_size = test_desc_batch.size(0)
            total_recall += batch_recall * batch_size
            total_samples += batch_size
            results.append(batch_recall)

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"[Evaluator] Processed {total_samples} samples, Current Recall@{k}: {total_recall/total_samples:.4f}")

        # Calculate final average recall
        avg_recall = total_recall / total_samples
        print(f"[Evaluator] Final Recall@{k}: {avg_recall:.4f}")

        return avg_recall, results

    def recall_at_k(self, query_desc, database_desc, k=10):
        """Compute recall@k metric for visual place recognition

        Args:
            query_desc (torch.Tensor): Query descriptors
            database_desc (torch.Tensor): Database descriptors
            k (int): Number of top matches to consider

        Returns:
            float: Recall@k score
        """
        # Convert database_desc list to tensor if needed
        if isinstance(database_desc, list):
            database_desc = torch.cat(database_desc, dim=0)

        # Compute similarity scores using cosine similarity
        similarity = torch.mm(query_desc, database_desc.t())

        # Get top k matches
        _, indices = similarity.topk(k=k, dim=1)

        # Get ground truth indices
        # Assuming sequential matching (query[i] should match database[i])
        gt_indices = torch.arange(len(query_desc)).cuda() if torch.cuda.is_available() else torch.arange(len(query_desc))

        # Check if ground truth is in top k predictions
        correct = 0
        for i, pred_indices in enumerate(indices):
            if gt_indices[i] in pred_indices:
                correct += 1

        # Calculate recall
        recall = correct / len(query_desc)

        return recall
