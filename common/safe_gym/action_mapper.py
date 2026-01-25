"""
This module maps action names to indices and vice-versa.
"""
from __future__ import annotations
import random
from typing import List, TYPE_CHECKING
import stormpy

if TYPE_CHECKING:
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
    def collect_actions(storm_bridge: StormBridge, analytical: bool = True) -> ActionMapper:
        """Collect all available actions.

        Args:
            storm_bridge (StormBridge): Reference to Storm Bridge
            analytical (bool): If True (default), use analytical approach that parses
                the PRISM program directly. If False, use simulation-based exploration.

        Returns:
            ActionMapper: Action Mapper
        """
        from common.safe_gym.storm_bridge import StormBridge as _StormBridge
        assert isinstance(storm_bridge, _StormBridge)

        if analytical:
            return ActionMapper.collect_actions_by_extraction(
                storm_bridge.path,
                storm_bridge.constant_definitions
            )

        # Simulation-based approach (legacy)
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

    @staticmethod
    def collect_actions_by_extraction(prism_path: str, constant_definitions: str = "") -> ActionMapper:
        """Collect all available actions by analyzing the PRISM program structure.

        This method extracts actions directly from the PRISM program without
        building or simulating the model. It iterates through all modules and
        commands to find action labels.

        Args:
            prism_path (str): Path to the PRISM file
            constant_definitions (str): Constant definitions for the PRISM file

        Returns:
            ActionMapper: Action Mapper with all actions from the PRISM program
        """
        assert isinstance(prism_path, str)
        assert isinstance(constant_definitions, str)

        action_mapper = ActionMapper()

        # Parse the PRISM program
        prism_program = stormpy.parse_prism_program(prism_path)

        # Preprocess with constant definitions
        prism_program = stormpy.preprocess_symbolic_input(
            prism_program, [], constant_definitions)[0].as_prism_program()

        # First pass: collect labeled actions and prepare suggestions for unlabeled
        suggestions = {}
        tau_counter = 0

        for module in prism_program.modules:
            for command in module.commands:
                if command.is_labeled:
                    # Labeled command - get the action name directly
                    action_mapper.add_action(command.action_name)
                else:
                    # Unlabeled command - will be assigned tau_X name
                    suggestions[command.global_index] = f"tau_{tau_counter}"
                    action_mapper.add_action(f"tau_{tau_counter}")
                    tau_counter += 1

        assert isinstance(action_mapper, ActionMapper)
        return action_mapper
