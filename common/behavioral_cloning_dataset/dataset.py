import stormpy
from typing import Any
import json
import numpy as np

class BehavioralCloningDataset:

    def __init__(self, config):
        self.config = config
        self.name = self.config.split(";")[0]
        self.prism_file = self.config.split(";")[1]
        self.prop = self.config.split(";")[2]
        self.constant_definitions = self.config.split(";")[3]

        

    def create(self, env):
        """
        Create the behavioral cloning dataset by interacting with the environment.

        This method should be implemented by subclasses to generate training data
        by collecting state-action pairs from expert demonstrations or optimal policies.

        Args:
            env: The environment to collect behavioral cloning data from.

        Returns:
            None. The dataset should be stored internally and accessible via get_data().
        """
        raise NotImplementedError

        

    def get_data(self) -> dict[str, Any]:
        """
        Retrieve the behavioral cloning dataset.

        Returns:
            A dictionary with keys 'X_train', 'y_train', 'X_test', 'y_test'.
            Test data may be None if unavailable.
        """
        raise NotImplementedError