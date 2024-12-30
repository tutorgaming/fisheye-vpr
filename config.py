#!/usr/bin/env python3
"""
Training Configuration Setting

"""
#####################################################################
# Imports
#####################################################################

#####################################################################
# Class
#####################################################################
class Configuration(object):
    """
    Metaclass for the configuration container
    """
    def __init__(self, config_path):
        self.config_path = config_path

    def select_dataset(
        self,
        config:dict = None
    ):
        """
        Select Dataset

        Args:
            config (dict, optional): _description_. Defaults to None.
        """

    def extract_config(self, config):
        dataset = select_dataset(config['dataset'])
        feature_extractor = select_feature_extractor(config['feature_extractor'])
        clustering = select_clustering(['clustering'])
        loss = select_loss(config['loss'])
        training = config['training']
        validation = config['validation']

        config_dict = {
            "dataset": dataset,
            "feature_extractor": feature_extractor,
            "clustering": clustering,
            "loss": loss,
            "training": training,
            "validation": validation,
        }

        return config_dict
