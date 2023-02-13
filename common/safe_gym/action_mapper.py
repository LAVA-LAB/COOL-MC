"""
This module maps action names to indices and vice-versa.
"""
from __future__ import annotations
import random
from typing import List
from common.safe_gym.storm_bridge import StormBridge

class ActionMapper:
    """
    The ActionMapper assigns each available action a unique action index [0,...]
    """

    def __init__(self):
        """
        Initialize ActionMapper.
        """
        self.actions = []

    def add_action(self, action_name: str):
        """Add action to action list to

        Args:
            action (str): Action name
        """
        if action_name not in self.actions:
            self.actions.append(action_name)
            self.actions.sort()

    def get_action_count(self):
        return len(self.actions)

    def action_index_to_action_name(self, nn_action_idx: int) -> str:
        """Action Index to action name.

        Args:
            nn_action_idx (int): Action index.

        Returns:
            str: Action name.
        """
        return self.actions[nn_action_idx]

    def action_name_to_action_index(self, action_name: str) -> int:
        """Action name to action index of

        Args:
            action_name (str): Action name

        Returns:
            int: Action index
        """
        assert isinstance(action_name, str)
        for i in range(len(self.actions)):
            if action_name == self.actions[i]:
                return i
        return None

    @staticmethod
    def collect_actions(storm_bridge: StormBridge) -> ActionMapper:
        """Collect all available actions.

        Args:
            storm_bridge (StormBridge): Reference to Storm Bridge

        Returns:
            ActionMapper: Action Mapper
        """
        assert isinstance(storm_bridge, StormBridge)
        action_mapper = ActionMapper()
        for epoch in range(50):
            storm_bridge.simulator.restart()
            for i in range(1000):
                actions = storm_bridge.simulator.available_actions()
                for action_name in actions:
                    # Add action if it is not in the list
                    action_mapper.add_action(str(action_name))
                # Choose randomly an action
                if storm_bridge.simulator.is_done():
                    break
                action_idx = random.randint(0, storm_bridge.simulator.nr_available_actions() - 1)
                storm_bridge.simulator.step(actions[action_idx])
        storm_bridge.simulator.restart()
        assert isinstance(action_mapper, ActionMapper)
        return action_mapper
