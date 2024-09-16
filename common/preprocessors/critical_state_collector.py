from common.preprocessors.preprocessor import Preprocessor
import numpy as np
import itertools

class CriticalStateCollector(Preprocessor):

    def __init__(self, state_mapper, config_str, task):
        super().__init__(state_mapper, config_str, task)
        self.name, self.critical_state_threshold = self.parse_config(self.config_str)
        self.all_critical_states = []
        self.collect = True
        self.critical_state = None
        

    def parse_config(self, config_str:str) -> None:
        """
        Parse the configuration.
        :param config_str: The configuration.
        """
        if len(config_str.split(';')) == 2:
            name, critical_state_threshold = config_str.split(';')
            critical_state_threshold = float(critical_state_threshold)
            return name, critical_state_threshold
        else:
            raise ValueError('Critical state must be specified')
        

    def preprocess(self, rl_agent, state:np.ndarray, action_mapper, current_action_name:str, deploy:bool) -> np.ndarray:
        """
        Perform the preprocessing.
        :param rl_agent: The RL agent
        :param current_action_name: The current action name during incremental building process
        :param state: The state.
        :param deploy: If the preprocessing is used during deployment
        :return: The preprocessed state.
        """
        self.state = state
        if not self.collect:
            # Later, when interpreting results
            return self.state
        # For collecting
        state_importance = float(rl_agent.q_eval.forward(state).max().item()-rl_agent.q_eval.forward(state).min().item())
        if rl_agent.q_eval.forward(state).max().item()-rl_agent.q_eval.forward(state).min().item()>self.critical_state_threshold:
            # Check if state is already in the list
            if not any(np.array_equal(state, cs[0]) for cs in self.all_critical_states):
                self.all_critical_states.append((state, state_importance))
                # Sort descending
                self.all_critical_states.sort(key=lambda x: x[1], reverse=True)
                print("Critical state found!")
            
        return self.state

    def force_true(self):
        # Check if state is equal to the critical state
        #print(self.state, self.critical_state)
        # Types
        #print(type(self.state), type(self.critical_state))
        # Length
        try:
            if self.critical_state != None:
                return np.array_equal(self.state, self.critical_state)
        except:
            return np.array_equal(self.state, self.critical_state)
        return False


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

