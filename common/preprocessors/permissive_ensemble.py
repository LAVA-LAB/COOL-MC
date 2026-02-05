"""
Permissive Ensemble Preprocessor.

This preprocessor enables all actions that any ensemble member would take,
creating a permissive policy that explores more states during model checking.
"""
from common.preprocessors.preprocessor import Preprocessor
import numpy as np


class PermissiveEnsemble(Preprocessor):
    """
    Preprocessor that makes an ensemble policy permissive.

    During model checking, this allows any action that at least one
    ensemble member would choose, enabling exploration of more states
    than the majority vote policy alone.
    """

    def __init__(self, state_mapper, config_str, task):
        super().__init__(state_mapper, config_str, task)
        self.parse_config(self.config_str)

    def parse_config(self, config_str: str) -> None:
        """
        Parse the configuration.

        Config format: permissive_ensemble
        (No additional parameters needed)
        """
        pass

    def preprocess(self, rl_agent, state: np.ndarray, action_mapper,
                   current_action_name: str, deploy: bool) -> np.ndarray:
        """
        Check if the current action is one that any ensemble member would take.

        Caches the allowed actions for should_allow_action to use.

        Args:
            rl_agent: The ensemble RL agent (must implement get_ensemble_actions)
            state: The current state
            action_mapper: Maps action indices to names
            current_action_name: The action being considered during model building
            deploy: Whether in deployment mode

        Returns:
            The state (possibly modified to influence action selection)
        """
        try:
            # Get all actions from ensemble members
            ensemble_actions = rl_agent.get_ensemble_actions(state)

            # Convert action indices to names (unique set)
            ensemble_action_names = list(set([
                action_mapper.action_index_to_action_name(a)
                for a in ensemble_actions
            ]))

            # Cache for should_allow_action
            self._current_allowed_actions = ensemble_action_names

            # If current action is in the ensemble's action set, allow it
            if current_action_name in ensemble_action_names:
                # Store in buffer that this action is allowed
                self.update_buffer(state, current_action_name, reset=False)
                return state

        except NotImplementedError:
            # Agent doesn't support ensemble actions, allow all
            self._current_allowed_actions = None
        except RuntimeError:
            # Agent not trained yet, allow all
            self._current_allowed_actions = None

        return state

    def get_allowed_actions(self, rl_agent, state: np.ndarray, action_mapper,
                            available_actions: list[str] = None) -> list[str]:
        """
        Get all action names that the ensemble would allow for this state.

        Args:
            rl_agent: The ensemble RL agent
            state: The current state
            action_mapper: Maps action indices to names
            available_actions: If provided, unavailable actions are replaced with
                available_actions[0] (matching model_checker fallback behavior)

        Returns:
            List of allowed action names
        """
        try:
            ensemble_actions = rl_agent.get_ensemble_actions(state)
            if available_actions:
                # Replace unavailable actions with fallback (same as model_checker)
                action_names = []
                for a in ensemble_actions:
                    name = action_mapper.action_index_to_action_name(a)
                    if name not in available_actions:
                        name = available_actions[0]
                    action_names.append(name)
                return list(set(action_names))
            else:
                return list(set([
                    action_mapper.action_index_to_action_name(a)
                    for a in ensemble_actions
                ]))
        except (NotImplementedError, RuntimeError):
            # Fall back to majority vote action only
            action_idx = rl_agent.select_action(state, deploy=True)
            return [action_mapper.action_index_to_action_name(action_idx)]

    def should_allow_action(self, state: np.ndarray, action_name: str) -> bool:
        """
        Check if the given action should be allowed for this state.

        This method is called by the model checker to restrict which actions
        are explored during induced model construction.

        Args:
            state: The current state (already preprocessed)
            action_name: The action name being considered

        Returns:
            True if the action is in the ensemble's permissive set, False otherwise
        """
        # Get allowed actions from the cache (set during preprocess)
        if hasattr(self, '_current_allowed_actions') and self._current_allowed_actions is not None:
            return action_name in self._current_allowed_actions
        return True  # If no cache or None, allow all actions

    def set_allowed_actions_for_state(self, rl_agent, state: np.ndarray, action_mapper, available_actions: list[str]):
        """
        Compute and cache the allowed actions for the current state.

        Called before should_allow_action to set up the permissive action set.

        Args:
            rl_agent: The ensemble RL agent
            state: The current state
            action_mapper: Maps action indices to names
            available_actions: List of actions available in the MDP at this state
        """
        try:
            ensemble_actions = rl_agent.get_ensemble_actions(state)
            # Replace unavailable actions with available_actions[0] (same fallback
            # as model_checker). This ensures the permissive set is a superset of
            # the majority vote action (which also uses this fallback).
            action_names = []
            for a in ensemble_actions:
                name = action_mapper.action_index_to_action_name(a)
                if name not in available_actions:
                    name = available_actions[0]
                action_names.append(name)
            self._current_allowed_actions = list(set(action_names))
        except (NotImplementedError, RuntimeError):
            # Fall back to all available actions
            self._current_allowed_actions = available_actions

    def save(self):
        """Saves the preprocessor in the MLFlow experiment."""
        pass

    def load(self, root_folder: str):
        """Loads the preprocessor from the folder."""
        pass
