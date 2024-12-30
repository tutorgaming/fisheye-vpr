#!/usr/bin/env python3
"""
Place Recognition Pipeline
- Create Database from the dataset
    - Global Descriptor Database
- Provide find function for finding N-candidates
- Provide the Evaluation Steps for the Place Recognition
"""
# Author: Theppasith N. <tutorgaming@gmail.com>
#####################################################################
# Imports
#####################################################################
import numpy as np
from database.generator import DatabaseGenerator
from pathlib import Path

#####################################################################
# Class
#####################################################################
class PlaceRecognitionModule(object):
    def __init__(self):
        # Load Global Descriptor Model
        self.model_path = Path()
        self.global_desc_model = None
        # Database Creator
        self.database_generator = DatabaseGenerator()

    def initialize(self):
        # Load the Database
        # Be Ready for the input
        pass

    def find_candidates(
        self,
        query_image: np.ndarray,
        num_cadidates:int = 10,
    )->dict:
        """
        Find the candidates from the database

        Args:
            query_image (_type_): _description_
        """
        result = {}
        return result