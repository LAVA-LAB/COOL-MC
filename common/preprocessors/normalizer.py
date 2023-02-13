from common.preprocessors.preprocessor import Preprocessor
import numpy as np
class Normalizer(Preprocessor):

    def __init__(self, state_mapper, config_str, task):
        super().__init__(state_mapper, config_str, task)
        self.denominator = self.parse_config(self.config_str)

    def parse_config(self, config_str:str) -> None:
        """
        Parse the configuration.
        :param config_str: The configuration.
        """
        return float(config_str.split(';')[1])

    def preprocess(self, rl_agent, state:np.ndarray, action_mapper, current_action_name:str, deploy:bool) -> np.ndarray:
        """
        Perform the preprocessing.
        :param rl_agent: The RL agent
        :param current_action_name: The current action name during incremental building process
        :param state: The state.
        :return: The preprocessed state.
        """
        return state / self.denominator

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

