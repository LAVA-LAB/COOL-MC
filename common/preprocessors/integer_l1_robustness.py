from common.preprocessors.preprocessor import Preprocessor
import numpy as np
from numpy import linalg as LA
import itertools

class IntegerL1Robustness(Preprocessor):

    def __init__(self, state_mapper, config_str, task):
        super().__init__(state_mapper, config_str, task)
        self.attack_name, self.epsilon, self.feature_indizes = self.parse_config(self.config_str)
        self.all_possible_attacks = None

    def parse_config(self, config_str:str) -> None:
        """
        Parse the configuration.
        :param config_str: The configuration.
        """
        if len(config_str.split(';')) == 2:
            attack_name, epsilon = config_str.split(';')
            feature_indizes = []
        else:
            attack_name, epsilon, feature_names = config_str.split(';')
            feature_names = feature_names.split(',')
            feature_indizes = []
            for feature_name in feature_names:
                feature_indizes.append(self.state_mapper.mapper[str(feature_name)])

        epsilon = float(epsilon)
        if epsilon.is_integer() == False:
            raise ValueError('Epsilon must be an integer')

        return attack_name, epsilon, feature_indizes

    def preprocess(self, rl_agent, state:np.ndarray, action_mapper, current_action_name:str, deploy:bool) -> np.ndarray:
        """
        Perform the preprocessing.
        :param rl_agent: The RL agent
        :param current_action_name: The current action name during incremental building process
        :param state: The state.
        :param deploy: If the preprocessing is used during deployment
        :return: The preprocessed state.
        """
        # Check if we generated all the possible attacks
        if self.in_buffer(state) == False:
            # If not, generate them and use the attack for the current action name
            adv_state_buffer = self.generate_attacks(rl_agent, state, action_mapper, *len(state)*[np.arange(-self.epsilon, self.epsilon+1)])
            self.update_buffer(state, adv_state_buffer)

        if current_action_name in self.buffer[str(state)].keys():
            return self.buffer[str(state)][current_action_name]
        else:
            return state

    def generate_attacks(self, rl_agent, state, action_mapper, *arrays):
        adv_state_buffer = {}
        original_action_idx = rl_agent.select_action(state, deploy=True)
        original_action_name = action_mapper.action_index_to_action_name(original_action_idx)
        adv_state_buffer[original_action_name] = state
        if self.all_possible_attacks == None:
            # Generate all possible l1-bounded attacks
            grid = np.meshgrid(*arrays)
            coord_list = [entry.ravel() for entry in grid]
            points = np.vstack(coord_list).T
            if len(self.feature_indizes) > 0:
                # Only for specific features
                for col in range(points.shape[1]):
                    target_feature = col in self.feature_indizes
                    if target_feature==False:
                        points=points[~(points[:,col] != 0),:]
            self.all_possible_attacks = list(points[np.absolute(points).sum(axis=1) <= self.epsilon,:])
        # Check all possible attacks and save the state if the action changes
        for attack in self.all_possible_attacks:
            adv_state = state+np.array(attack)
            action_idx = rl_agent.select_action(adv_state,deploy=True)
            action_name = action_mapper.action_index_to_action_name(action_idx)
            if action_name not in adv_state_buffer.keys():
                adv_state_buffer[action_name] = np.array(adv_state)
            if len(adv_state_buffer.keys()) == len(action_mapper.actions):
                break
        return adv_state_buffer

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

