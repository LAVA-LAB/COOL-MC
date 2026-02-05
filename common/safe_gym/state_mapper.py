"""
This module contains the StateMapper.
"""
import json
import os
import hashlib
from typing import Union
import numpy as np


class StateMapper:
    """The StateMapper maps the state at the right position in the state
    numpy state array. It is needed in the case of state variables which
    are disabled and in the case when the PRISM state space has not the
    same format as the OpenAI gym state space.
    """

    def __init__(self, prism_file_path: str, transformation_file_path: str,
                 state_json_example: str, disabled_features: str):
        """Initialize StateMapper

        Args:
            prism_file_path (str): Full path to the PRISM file
            transformation_file_path (str): Transformation file path
            state_json_example (str): state json example from StormBridge
            disabled_features (str): Disabled features and seperated by commata
        """
        assert isinstance(prism_file_path, str)
        assert isinstance(transformation_file_path, str)
        assert isinstance(state_json_example, str)
        assert isinstance(disabled_features, str)

        # Store the PRISM filename
        self.prism_filename = os.path.basename(prism_file_path)

        # Check if compressed state representation folder exists
        prism_dir = os.path.dirname(prism_file_path)
        # Remove .prism extension for folder name (e.g., "csma.2-2.v1.prism" -> "csma.2-2.v1")
        prism_name_without_ext = self.prism_filename.replace('.prism', '')
        self.compressed_folder_path = os.path.join(prism_dir, prism_name_without_ext)
        self.compressed_state_representation = os.path.isdir(self.compressed_folder_path)

        # Load and cache compressed feature names if available
        self.compressed_feature_names = None
        if self.compressed_state_representation:
            try:
                self.compressed_feature_names = self.load_compressed_feature_names()
            except (FileNotFoundError, IOError):
                # If feature_names.txt doesn't exist, keep as None
                pass

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
        # Infer dtype from input state to support both integers and floats
        mapped_state = np.zeros(size, dtype=state.dtype)
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

    def state_to_str(self, state: np.ndarray) -> str:
        """Convert mapped state to string representation with format 'FEATURE_NAME=VALUE'.

        Args:
            state (np.ndarray): Mapped state (output from map() method)

        Returns:
            str: String representation with format 'FEATURE1=VALUE1,FEATURE2=VALUE2,...'
        """
        assert isinstance(state, np.ndarray)
        feature_strings = []
        for name, idx in sorted(self.mapper.items(), key=lambda x: x[1]):
            value = state[idx]
            feature_strings.append(f"{name}={value}")
        result = ','.join(feature_strings)
        assert isinstance(result, str)
        return result

    def decompress_state(self, state_str: str) -> np.ndarray:
        """Load compressed state from file using SHA256 hash of state string as filename.

        Args:
            state_str (str): State string (e.g., 'x=5,y=10' from state_to_str())

        Returns:
            np.ndarray: Decompressed state array loaded from .npy file

        Raises:
            FileNotFoundError: If the compressed state file doesn't exist
            ValueError: If compressed state representation is not available
        """
        assert isinstance(state_str, str)

        if not self.compressed_state_representation:
            raise ValueError(
                f"Compressed state representation not available for {self.prism_filename}. "
                f"Folder {self.compressed_folder_path} does not exist."
            )

        # Create SHA256 hash of the state string
        hash_value = hashlib.sha256(state_str.encode()).hexdigest()

        # Create file path: compressed_folder/hash.npy
        file_path = os.path.join(self.compressed_folder_path, f"{hash_value}.npy")

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Compressed state file not found: {file_path} (hash of '{state_str}')"
            )

        # Load and return the numpy array
        decompressed_state = np.load(file_path)
        assert isinstance(decompressed_state, np.ndarray)
        return decompressed_state

    def load_compressed_feature_names(self) -> list:
        """Load feature names from the compressed state representation folder.

        Reads the feature_names.txt file from the compressed folder where each
        line contains one feature name.

        Returns:
            list: List of feature names from the compressed representation

        Raises:
            FileNotFoundError: If the feature_names.txt file doesn't exist
            ValueError: If compressed state representation is not available
        """
        if not self.compressed_state_representation:
            raise ValueError(
                f"Compressed state representation not available for {self.prism_filename}. "
                f"Folder {self.compressed_folder_path} does not exist."
            )

        # Create file path: compressed_folder/feature_names.txt
        file_path = os.path.join(self.compressed_folder_path, "feature_names.txt")

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Feature names file not found: {file_path}"
            )

        # Read feature names line by line
        feature_names = []
        with open(file_path, 'r') as f:
            for line in f:
                # Strip whitespace and newlines from each line
                feature_name = line.strip()
                if feature_name:  # Skip empty lines
                    feature_names.append(feature_name)

        return feature_names

    def get_compressed_feature_names(self) -> Union[list, None]:
        """Get cached compressed feature names.

        Returns the cached compressed feature names that were loaded during
        initialization. If compressed state representation is not available
        or feature_names.txt doesn't exist, returns None.

        Returns:
            list or None: List of feature names from compressed representation,
                         or None if not available
        """
        return self.compressed_feature_names

    def compressed_inverse_mapping(self, idx: int) -> Union[str, None]:
        """Map index to compressed feature name.

        Maps an index in the decompressed state array to the corresponding
        feature name from the compressed representation.

        Args:
            idx (int): Feature index in decompressed state array

        Returns:
            str or None: Feature name at the given index, or None if index
                        is out of bounds or compressed features not available
        """
        if self.compressed_feature_names is None:
            return None

        if 0 <= idx < len(self.compressed_feature_names):
            return self.compressed_feature_names[idx]

        return None


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

    def get_prism_filename(self) -> str:
        """Get the PRISM filename.

        Returns:
            str: PRISM filename
        """
        return self.prism_filename

    def has_compressed_state_representation(self) -> bool:
        """Check if compressed state representation folder exists.

        Returns:
            bool: True if compressed state representation folder exists
        """
        return self.compressed_state_representation
