from common.preprocessors.preprocessor import Preprocessor
import numpy as np

class PolicyAbstraction(Preprocessor):

    def __init__(self, state_mapper, config_str, task):
        super().__init__(state_mapper, config_str, task)
        self.parse_config(self.config_str)


    def parse_config(self, config_str:str) -> None:
        """
        Parse the configuration.
        :param config_str: The configuration.

        Example: policy_abstraction;fuel=[0,2,4,6]
        """
        self.feature_mapper = {}
        for feature in config_str.split(";")[1:]:
            feature_name, values = feature.split("=")
            values = values.replace("[", "").replace("]", "").split(",")
            values = [int(x) for x in values]
            self.feature_mapper[feature_name] = values

    def get_closest_lower_bound(self, feature_values, current_value):
        """
        Map the current value to the closest value in the feature_values list.

        Parameters:
            feature_values (list): A list of values to map the current value to.
            current_value (float or int): The current value to map to a value in feature_values.

        Returns:
            closest_value (float or int): The closest value in feature_values to current_value.
        """
        closest_value = min(feature_values, key=lambda x: abs(x - current_value))
        return closest_value

    def get_closest_upper_bound(self, feature_values, current_value):
        """
        Map the current value to the closest upper bound value in the feature_values list.

        Parameters:
            feature_values (list): A list of values to map the current value to.
            current_value (float or int): The current value to map to a value in feature_values.

        Returns:
            closest_value (float or int): The closest upper bound value in feature_values to current_value.
        """
        upper_bounds = [val for val in feature_values if val > current_value]
        if len(upper_bounds) > 0:
            closest_value = min(upper_bounds, key=lambda x: abs(x - current_value))
        else:
            closest_value = None
        return closest_value


    def preprocess(self, rl_agent, state:np.ndarray, action_mapper, current_action_name:str, deploy:bool) -> np.ndarray:
        """
        Perform the preprocessing.
        :param rl_agent: The RL agent
        :param current_action_name: The current action name during incremental building process
        :param state: The state.
        :return: The preprocessed state.
        """
        original_state = state.copy()
        for i, value in enumerate(list(original_state)):
            # Feature index to feature name
            feature_name = self.state_mapper.inverse_mapping(i)
            if feature_name in self.feature_mapper.keys():
                lower_bound = self.get_closest_lower_bound(self.feature_mapper[feature_name], original_state[i])
                if lower_bound > original_state[i]:
                    return original_state
                upper_bound = self.get_closest_upper_bound(self.feature_mapper[feature_name], original_state[i])
                # If is true, if upper_bound is not None and upper_bound < original_state[i]
                if upper_bound is None:
                    return original_state
                else:
                    for v in range(lower_bound, upper_bound):
                        tmp_state = state.copy()
                        tmp_state[i] = v
                        # Get action
                        action_idx = rl_agent.select_action(tmp_state, deploy)
                        action_name = action_mapper.action_index_to_action_name(action_idx)
                        if action_name == current_action_name:
                            return tmp_state
        return original_state


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

