"""
This module contains the Safe Gym.
"""
import math
from typing import Tuple, Union
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Discrete
import numpy as np
from common.safe_gym.storm_bridge import StormBridge
from common.safe_gym.action_mapper import ActionMapper


class SafeGym(gym.Env):
    """
    The SafeGym environment is the interface between the Storm Simulator and
    the RL agent. It behaves as the OpenAI gym.
    """
    def __init__(self, prism_file_path: str, constant_definitions: str, max_steps: int,
                 wrong_action_penalty: int, reward_flag: bool, seed: int,
                 disabled_features: str = ''):
        """Initialize the SafeGym. The SafeGym needs the information about the PRISM environment.

        Args:
            prism_file_path (str): The Path to the PRISM environment
            constant_definitions (str): The constant definitions for the PRISM environment
            max_steps (int): The number of maximal steps in the environment
            wrong_action_penalty (int): The penalty for wrong actions
            reward_flag (bool): If true, rewards.
                                    otherwise penalties
            seed (int): random seed
            disabled_features (str, optional): [description]. Disabled features/state-variables
                                (seperated by commatas. Defaults to ''.
        """
        assert isinstance(prism_file_path, str)
        assert isinstance(constant_definitions, str)
        assert isinstance(wrong_action_penalty, int)
        assert isinstance(reward_flag, bool)
        assert isinstance(disabled_features, str)
        assert isinstance(seed, int)
        self.storm_bridge = StormBridge(prism_file_path, constant_definitions, wrong_action_penalty,
                                        reward_flag, disabled_features, seed)
        self.action_mapper = ActionMapper.collect_actions(self.storm_bridge)
        self.steps = 0
        self.max_steps = max_steps
        self.state = self.reset()
        # Observation Space
        self.observation_space = spaces.Box(np.array(
            [-math.inf]*self.state.shape[0]), np.array([math.inf]*self.state.shape[0]))
        # Action Space
        self.action_space = Discrete(len(self.action_mapper.actions))

    def step(self, action_index: int) -> Tuple[np.ndarray, Union[float, int], bool, dict]:
        """Executes the action passed via the action_index (int).

        Args:
            action_index ([int]): Action INdex

        Returns:
            [Tuple]: (State, Reward, done, info)
        """
        assert isinstance(action_index, int)
        action_name = self.action_mapper.action_index_to_action_name(
            action_index)
        n_state, reward, self.done = self.storm_bridge.step(action_name)
        self.steps += 1
        if self.steps >= self.max_steps:
            self.truncated = True
            self.done = True
        self.state = n_state
        assert isinstance(self.state, np.ndarray)
        assert isinstance(reward, float) or isinstance(reward, int)
        assert isinstance(self.done, bool)
        return self.state, reward, self.done, self.truncated, {}

    def reset(self) -> np.ndarray:
        """Resets the Gym

        Returns:
            [Numpy Array]: Init state
        """
        self.steps = 0
        self.state = self.storm_bridge.reset()
        self.done = False
        self.truncated = False
        assert isinstance(self.state, np.ndarray)
        return self.state
