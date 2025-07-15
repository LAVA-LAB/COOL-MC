"""
Storm Bridge module.
This module builds the bridge between COOL-MC and Storm.
"""
import os
import json
from typing import Tuple, Union
from aenum import constant
import numpy as np
import stormpy.simulator
import stormpy.examples.files
import stormpy.utility
from common.safe_gym.model_checker import ModelChecker
from common.safe_gym.state_mapper import StateMapper
from stormpy.utility.utility import JsonContainerDouble
from stormpy.simulator import PrismSimulator


class StormBridge:
    """
    This class connects COOL-MC with Storm.
    It is the only class that has contact with Storm.
    """

    def __init__(self, path: str, constant_definitions: str, wrong_action_penalty: int,
                 reward_flag: bool, disabled_features: str, seed: int):
        """
        Initialize the Storm Bridge.

        Args:
            path (str): Path to the PRISM-file
            constant_definitions (str): Constant definitions for the PRISM-file
            wrong_action_penalty (int): The RL agent penalty for choosing a wrong action.
            reward_flag (bool): if True, environment is working with rewards.
                                    If False, enviornment works with penalty.
            disabled_features (str): Defines which features of the MDP should
                                        be invisible for the RL agent.
            seed (int): Seed for the simulator.
        """
        assert isinstance(path, str)
        assert isinstance(constant_definitions, str)
        assert isinstance(wrong_action_penalty, int)
        assert isinstance(reward_flag, bool)
        assert isinstance(disabled_features, str)
        assert isinstance(seed, int)
        self.seed = seed
        self.disabled_features = disabled_features
        self.simulator = self.create_simulator(path, constant_definitions)
        self.constant_definitions = constant_definitions
        self.state_json_example = self.__preprocess_state_json_example(
            self.simulator.restart()[0])
        self.wrong_action_penalty = wrong_action_penalty
        self._state = None
        self.reward = None
        self.reward_flag = reward_flag
        self.path = path
        json_path = os.path.splitext(self.path)[0]+'.json'
        self.state_mapper = StateMapper(
            json_path, self.state_json_example, self.disabled_features)
        self.model_checker = ModelChecker(self.state_mapper)

    def __preprocess_state_json_example(self, json_example: JsonContainerDouble) -> str:
        """Preprocess the state by casting boolean values to int values.

        Args:
            json_example (stormpy.utility.utility.JsonContainerDouble): State

        Returns:
            str: State as JSON string
        """
        assert isinstance(
            json_example, stormpy.utility.utility.JsonContainerDouble)
        dummy_state = {}
        json_example = str(json_example)
        for k in json.loads(json_example):
            value = json.loads(json_example)[k]
            if isinstance(value, bool):
                if value:
                    value = 1
                else:
                    value = 0
            dummy_state[k] = int(value)
        json_object = json.dumps(dummy_state)
        assert isinstance(json_object, str)
        return json_object

    def create_simulator(self, path: str, constant_definitions: str) -> PrismSimulator:
        """Create the simulator for PRISM file path and constant definitions.

        Args:
            path (str): Path to the PRISM file
            constant_definitions (str): Constant definitions for PRISM file

        Returns:
            PrismSimulator: PRISM simulator
        """
        assert isinstance(path, str)
        assert isinstance(constant_definitions, str)
        prism_program = stormpy.parse_prism_program(path)
        prism_program = stormpy.preprocess_symbolic_input(
            prism_program, [], constant_definitions)[0].as_prism_program()
        suggestions = dict()
        i = 0
        for module in prism_program.modules:
            for command in module.commands:
                if not command.is_labeled:
                    suggestions[command.global_index] = "tau_" + \
                        str(i)  # str(m.name)
                    i += 1

        prism_program = prism_program.label_unlabelled_commands(suggestions)
        if self.seed != -1:
            simulator = stormpy.simulator.create_simulator(
                prism_program, seed=self.seed)
        else:
            simulator = stormpy.simulator.create_simulator(prism_program)
        simulator.set_action_mode(
            stormpy.simulator.SimulatorActionMode.GLOBAL_NAMES)
        assert isinstance(simulator, PrismSimulator)
        return simulator

    def step(self, action_name: str) -> Tuple[np.ndarray, Union[float, int], bool]:
        """Executes a step in the simulator based on the given
        action.

        Args:
            action_name (str): Action name.

        Returns:
            Tuple[np.ndarray, float, bool]: Returns new state, reward, and if simulator done.
        """
        assert isinstance(action_name, str)
        penalty = 0
        current_available_actions = sorted(self.simulator.available_actions())
        if len(current_available_actions) == 0:
            return self._state, self.reward, True
        if action_name not in current_available_actions:
            action_name = current_available_actions[0]
            penalty = -self.wrong_action_penalty


        data = self.simulator.step(action_name)
        
        self._state = str(data[0])
        self.reward = data[1]


        self._state = self.parse_state(self._state)
        self.reward = self.reward[0]
        if self.reward_flag is not True:
            self.reward *= (-1)
        if penalty != 0:
            self.reward = penalty
        done = self.simulator.is_done()
        assert isinstance(self._state, np.ndarray)
        assert isinstance(self.reward, float) or isinstance(self.reward, int)
        assert isinstance(done, bool)
        return self._state, self.reward, done

    def reset(self) -> np.ndarray:
        """Resets the Storm simulator.

        Returns:
            np.ndarray: Storm simulator state.
        """
        data = self.simulator.restart()
        _state = str(data[0])
        self._state = self.parse_state(str(_state))
        assert isinstance(self._state, np.ndarray)
        return self._state

    def parse_state(self, state: str) -> np.ndarray:
        """Parse json state to numpy states

        Args:
            state (np.ndarray): [description]

        Returns:
            [type]: [description]
        """
        assert isinstance(state, str)
        # State Extracting
        arr = []
        state_variables = []
        for k in json.loads(state):
            value = json.loads(state)[k]
            if isinstance(value, bool):
                if value:
                    value = 1
                else:
                    value = 0
            state_variables.append(k)
            arr.append(value)
        state = np.array(arr, dtype=np.int32)
        # Mapping and deleting disabled features
        state = self.state_mapper.map(state)
        state = np.array(state, dtype=np.int32)
        assert isinstance(state, np.ndarray)
        return state
