#!/usr/bin/env python3

import sys
import unittest
import json
import torch
from pathlib import Path
# Path for Import Internal Modules
sys.path.append("/workspace/fisheye-vpr")
from evaluator import Evaluator
from dataloaders.isaac_office_all_fisheye import IsaacOffice_All_Fisheye_Dataset
from main import VPR

class TestEvaluator(unittest.TestCase):
    def setUp(self):
        # Setup test data paths
        self.result_path = Path("/workspace/old_results/20250102/resnet18finetuned_NetVLAD64_IsaacOfficeAll_Fisheye_02-Jan-2025_21-40-49")
        self.result_json = self.result_path / "result.json"

        # Load results and config
        self.results = self.load_config(self.result_json)

        # Initialize dataset
        self.dataset = IsaacOffice_All_Fisheye_Dataset()

        # Get best model path
        self.best_model_path = self.results["best_model_train_path"]

        # Initialize VPR model with original config
        self.vpr = VPR(self.results)

        # Load best checkpoint
        checkpoint = torch.load(self.best_model_path)
        self.vpr.model.load_state_dict(checkpoint)

        # Create evaluator
        self.evaluator = Evaluator(
            model=self.vpr.model.to("cuda"),
            dataset=self.dataset
        )

    def load_config(self, result_file_path):
        try:
            with open(result_file_path, "r") as f:
                # Using ast.literal_eval to safely evaluate string representation of dict
                import ast
                config_str = f.read()
                loaded_config = ast.literal_eval(config_str)
                return loaded_config
        except FileNotFoundError:
            print(f"Config file not found at {result_file_path}")
            return None
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing config file: {e}")
            return None

    def test_evaluation(self):
        # Run evaluation
        recalls = []
        # recall_k = 10 # Test with recall@1
        for k in range(1,11):
            avg_recall, results = self.evaluator.evaluate(k=k)
            recalls.append(avg_recall)

            # Basic assertions
            self.assertIsInstance(avg_recall, float)
            self.assertGreaterEqual(avg_recall, 0.0)
            self.assertLessEqual(avg_recall, 1.0)

            # Check results list
            self.assertIsInstance(results, list)
            self.assertGreater(len(results), 0)

            # print(f"Average Recall@{k}: {avg_recall}")

        for k, recall_at_k in enumerate(recalls):
            print(f"Recall@{k+1} = {recall_at_k}")

if __name__ == '__main__':
    unittest.main()