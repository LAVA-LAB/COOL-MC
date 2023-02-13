"""
This module contains the StateMapper.
"""
import json
import os
from typing import Union
import numpy as np


class StateMapper:
    """The StateMapper maps the state at the right position in the state
    numpy state array. It is needed in the case of state variables which
    are disabled and in the case when the PRISM state space has not the
    same format as the OpenAI gym state space.
    """

    def __init__(self, transformation_file_path: str,
                 state_json_example: str, disabled_features: str):
        """Initialize StateMapper

        Args:
            transformation_file_path (str): Transformation file path
            state_json_example (str): state json example from StormBridge
            disabled_features (str): Disabled features and seperated by commata
        """
        assert isinstance(transformation_file_path, str)
        assert isinstance(state_json_example, str)
        assert isinstance(disabled_features, str)
        self.mapper = self.load_mappings(
            transformation_file_path, state_json_example)
        self.original_format = list(self.mapper)
        if disabled_features == "":
            self.disabled_features = []
        else:
            self.disabled_features = disabled_features.split(",")
        self.mapper = self.update_mapper(self.mapper, self.disabled_features)

    def update_mapper(self, mapper: dict, disabled_features: list) -> dict:
        """Update mapper based on disabled features.

        Args:
            mapper (dict): Mapper
            disabled_features (str): Disabled features seperated by commatas

        Returns:
            dict: Mapper
        """
        assert isinstance(mapper, dict)
        assert isinstance(disabled_features, list)
        if len(disabled_features) == 0:
            return mapper
        for disabled_key in disabled_features:
            disabled_index = mapper[disabled_key]
            for key in mapper.keys():
                if key != disabled_key and mapper[key] > disabled_index:
                    mapper[key] -= 1
            del mapper[disabled_key]
        assert isinstance(mapper, dict)
        return mapper

    def load_mappings(self, transformation_file_path: str, state_json_example: str) -> dict:
        """Loads the mapping from the transformation file or from the state JSON example.

        Args:
            transformation_file_path (str): Transformation path or transformation parameter
            state_json_example (str): State JSON example

        Returns:
            dict: state variable mapper
        """
        mapper = None
        if os.path.exists(transformation_file_path):
            with open(transformation_file_path) as json_file:
                mapper = json.load(json_file)
        else:
            json_example = str(state_json_example)
            mapper = {}
            i = 0
            for k in json.loads(json_example):
                mapper[k] = i
                i += 1

        return mapper

    def map(self, state: np.ndarray) -> np.ndarray:
        """Maps the state variables of the given state into the
        correct format.

        Args:
            state (np.ndarray): Raw state

        Returns:
            np.ndarray: Transformed state
        """
        assert isinstance(state, np.ndarray)
        size = len(self.mapper.keys())
        mapped_state = np.zeros(size, np.int32)
        # print(state)
        for idx, name in enumerate(self.original_format):
            if name not in self.disabled_features:
                n_idx = self.mapper[name]
                mapped_state[n_idx] = state[idx]
        assert isinstance(mapped_state, np.ndarray)
        return mapped_state

    def get_feature_names(self):
        feature_names = []
        for idx, name in enumerate(self.original_format):
            if name not in self.disabled_features:
                feature_names.append(name)
        return feature_names


    def inverse_mapping(self, idx: int) -> Union[str, None]:
        """Map index to state variable name.

        Args:
            idx (int): State variable index.

        Returns:
            str: State variable name
        """
        for name in self.mapper:
            try:
                if self.mapper[name] == idx:
                    return name
            except:
                pass
        return None

    def get_state_size(self):
        return len(self.mapper.keys())
