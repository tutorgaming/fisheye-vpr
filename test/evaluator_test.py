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
        self.result_path = Path("/workspace/results/20250101/resnet18_NetVLAD64_IsaacOfficeAll_Fisheye_01-Jan-2025_11-50-31")
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
        self.vpr.model.load_state_dict(checkpoint["model_state_dict"])

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
        recall_k = 1 # Test with recall@1
        avg_recall, results = self.evaluator.evaluate(k=recall_k)

        # Basic assertions
        self.assertIsInstance(avg_recall, float)
        self.assertGreaterEqual(avg_recall, 0.0)
        self.assertLessEqual(avg_recall, 1.0)

        # Check results list
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)

        print(f"Average Recall@{recall_k}: {avg_recall}")

if __name__ == '__main__':
    unittest.main()