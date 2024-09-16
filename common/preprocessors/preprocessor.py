import numpy as np

class Preprocessor:

    def __init__(self, state_mapper, config_str, task):
        self.state_mapper = state_mapper
        self.config_str = config_str
        self.task = task
        self.buffer = {}


    def parse_config(self, config_str:str) -> None:
        """
        Parse the configuration.
        :param config_str: The configuration.
        """
        raise NotImplementedError()

    def preprocess(self, rl_agent, state:np.ndarray, action_mapper, current_action_name:str, deploy:bool) -> np.ndarray:
        """
        Perform the preprocessing.
        :param rl_agent: The RL agent
        :param current_action_name: The current action name during incremental building process
        :param state: The state.
        :return: The preprocessed state.
        """
        raise NotImplementedError()

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

    def update_buffer(self, state, value, reset=True):
        if reset:
            self.buffer = {}
        self.buffer[str(state)] = value

    def in_buffer(self, state: np.ndarray) -> bool:
        if str(state) in self.buffer.keys():
            return True
        return False

    def force_true(self):
        return False

