import numpy as np

class ActionRobustness:

    def __init__(self, state_mapper, config_str, task):
        self.state_mapper = state_mapper
        self.config_str = config_str
        self.task = task
        self.parse_config(config_str)
        self.buffer = {}


    def parse_config(self, config_str:str) -> None:
        """
        Parse the configuration.
        :param config_str: The configuration.
        """
        self.top_n = int(config_str.split(";")[1])


    def preprocess(self, rl_agent, state:np.ndarray, action_mapper, current_action_name:str, deploy:bool) -> np.ndarray:
        """
        Perform the preprocessing.
        :param rl_agent: The RL agent
        :param current_action_name: The current action name during incremental building process
        :param state: The state.
        :return: The preprocessed state.
        """
        # Get the top N actions:
        q_values = rl_agent.q_eval.forward(state)
        # q_values to numpy
        q_values = q_values.detach().numpy()
      
        
        # Get the top N actions
        top_n_actions = np.argsort(q_values)[-self.top_n:]
    

        # to 1d list
        top_n_actions = top_n_actions.tolist()
        # Get the action name
        self.force_true_value = False
        for action_idx in top_n_actions:
            action_name = action_mapper.action_index_to_action_name(action_idx)
            if action_name == current_action_name:
                self.force_true_value = True
                return state
        
        self.force_true_value = False
        
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

    def update_buffer(self, state, value, reset=True):
        if reset:
            self.buffer = {}
        self.buffer[str(state)] = value

    def in_buffer(self, state: np.ndarray) -> bool:
        if str(state) in self.buffer.keys():
            return True
        return False

    def force_true(self):
        # Force for eacht state the top N actions to be true
        return self.force_true_value

