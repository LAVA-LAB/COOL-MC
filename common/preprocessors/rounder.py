from common.preprocessors.preprocessor import Preprocessor
import numpy as np
class Rounder(Preprocessor):

    def __init__(self, state_mapper, config_str, task):
        super().__init__(state_mapper, config_str, task)
        self.attack_name, self.rounding_type = self.parse_config(self.config_str)

    def parse_config(self, config_str:str) -> None:
        """
        Parse the configuration.
        :param config_str: The configuration.
        """
        attack_name, rounding_type = config_str.split(';')
        return attack_name, rounding_type

    def preprocess(self, rl_agent, state:np.ndarray, action_mapper, current_action_name:str, deploy:bool) -> np.ndarray:
        """
        Perform the preprocessing.
        :param rl_agent: The RL agent
        :param current_action_name: The current action name during incremental building process
        :param state: The state.
        :return: The preprocessed state.
        """
        # rounds to the nearest integer
        if self.rounding_type == 'round':
            return np.round(state)
        elif self.rounding_type == 'floor':
            return np.floor(state)
        else:
            raise ValueError('Rounding type not supported')

    def save(self):
        """
        Saves the preprocessor in the MLFlow experiment.
        """
        pass

    def load(self, root_folder:str):
        """
        Loads the preprocessor from the folder

        Args:
            root_folder ([str]): Path to the folder
        """
        pass

