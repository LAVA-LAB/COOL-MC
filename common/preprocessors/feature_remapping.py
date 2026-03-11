from common.preprocessors.preprocessor import Preprocessor
import numpy as np

class FeatureRemapper(Preprocessor):

    def __init__(self, state_mapper, config_str, task):
        super().__init__(state_mapper, config_str, task)
        self.parse_config(self.config_str)


    def parse_config(self, config_str:str) -> None:
        """
        Parse the configuration.
        :param config_str: The configuration.

        Closest-value mode:  feature_remapping;fuel=[0,2,4,6]
        Explicit dict mode:  feature_remapping;year={0:16,1:17,2:18,3:19}
        """
        self.feature_mapper = {}
        self.feature_dict_mapper = {}
        for feature in config_str.split(";")[1:]:
            feature_name, feature_values = feature.split("=", 1)
            if feature_values.startswith("{") and feature_values.endswith("}"):
                # Explicit dict mapping: {0:16,1:17,2:18,3:19}
                inner = feature_values[1:-1]
                mapping = {}
                for pair in inner.split(","):
                    k, v = pair.split(":")
                    mapping[int(k.strip())] = int(v.strip())
                self.feature_dict_mapper[feature_name] = mapping
            else:
                # Closest-value list: [0,2,4,6]
                feature_values = feature_values.replace("[", "").replace("]", "").split(",")
                feature_values = [int(x) for x in feature_values]
                self.feature_mapper[feature_name] = feature_values

    def feature_remapping(self, feature_values, current_value):
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

    def preprocess(self, rl_agent, state:np.ndarray, action_mapper, current_action_name:str, deploy:bool) -> np.ndarray:
        """
        Perform the preprocessing.
        :param rl_agent: The RL agent
        :param current_action_name: The current action name during incremental building process
        :param state: The state.
        :return: The preprocessed state.
        """
        if self.in_buffer(state):
            return self.buffer[str(state)]
        else:
            original_state = state.copy()
            for i, value in enumerate(list(original_state)):
                # Feature index to feature name
                feature_name = self.state_mapper.inverse_mapping(i)
                if feature_name in self.feature_dict_mapper:
                    key = int(original_state[i])
                    if key in self.feature_dict_mapper[feature_name]:
                        state[i] = self.feature_dict_mapper[feature_name][key]
                elif feature_name in self.feature_mapper:
                    state[i] = self.feature_remapping(self.feature_mapper[feature_name], original_state[i])
                self.update_buffer(original_state, state, True)
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

