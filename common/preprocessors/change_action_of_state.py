from common.preprocessors.preprocessor import Preprocessor
import numpy as np
import time

class ChangeActionOfState(Preprocessor):

    def __init__(self, state_mapper, config_str, task):
        super().__init__(state_mapper, config_str, task)
        self.state, self.action = self.parse_config(self.config_str)
        self.adv_state = None


    def parse_config(self, config_str:str) -> None:
        """
        Parse the configuration.
        :param config_str: The configuration.
        """
        state = np.fromstring(config_str.split(';')[1][1:-1], sep=' ', dtype=int)
        action = int(config_str.split(';')[2])
        return state, action

    def find_correct_manipulation(self, agent):
        shape = np.shape(self.state)
        #print("Find correct manipulation")
        # Create random array with the same dimensions
        start_time = time.time()
        while True:
            random_array = np.random.randint(low=0, high=10, size=shape)
            #print(random_array, agent.select_action(random_array, True), self.action)
            if agent.select_action(random_array, True) == self.action:
                #print("Time to find correct manipulation: ", time.time() - start_time)
                return random_array

        return random_array



    def preprocess(self, rl_agent, state:np.ndarray, action_mapper, current_action_name:str, deploy:bool) -> np.ndarray:
        """
        Perform the preprocessing.
        :param rl_agent: The RL agent
        :param current_action_name: The current action name during incremental building process
        :param state: The state.
        :return: The preprocessed state.
        """
        if np.array_equal(state, self.state):
            try:
                if self.adv_state == None:
                    self.adv_state = self.find_correct_manipulation(rl_agent)
            except:
                pass
            return self.adv_state
        else:
            return state


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

