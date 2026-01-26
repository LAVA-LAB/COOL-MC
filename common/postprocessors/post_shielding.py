from common.postprocessors.postprocessor import Postprocessor
from common.behavioral_cloning_dataset.behavioral_cloning_dataset_builder import *
import numpy as np

class PostShielding(Postprocessor):

    def __init__(self, state_mapper, config):
        super().__init__(state_mapper, config)
        # Config format: post_shielding;dataset_type;prism_file;property;constant_definitions
        # Extract everything after "post_shielding;" and rejoin for the dataset builder
        self.dataset_config = ";".join(config.split(";")[1:])
        self.dataset = BehavioralCloningDatasetBuilder.build(self.dataset_config)
        self.dataset.created = False


    def postprocess_before_step(self, rl_agent, state, action):
        if self.dataset != None and self.dataset.created == False:
            self.dataset.create(self.env)
            self.dataset.created = True

        # Find all optimal actions for current state
        matching_indices = np.where((self.dataset.X == state).all(axis=1))[0]
        if len(matching_indices) > 0:
            optimal_actions = self.dataset.y[matching_indices]
            # Check if action is in optimal actions
            if action not in optimal_actions:
                # Replace with random optimal action from dataset
                action = int(np.random.choice(optimal_actions))

        return action


    def save(self):
        """
        Saves the manipulator in the MLFlow experiment.
        """
        pass

    def load(self, root_folder:str):
        """
        Loads the manipulator from the folder

        Args:
            root_folder ([str]): Path to the folder
        """
        pass

