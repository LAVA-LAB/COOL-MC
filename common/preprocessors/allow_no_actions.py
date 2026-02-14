import numpy as np
from common.preprocessors.preprocessor import Preprocessor


class AllowNoActions(Preprocessor):
    """Preprocessor that disallows all actions at every state.

    When used during model checking, this causes the model checker to reject
    all actions, effectively making every state an absorbing/terminal state.
    Useful for baseline analyses (e.g., "what happens with no treatment").

    Configuration: "allow_no_actions"
    """

    def __init__(self, state_mapper, config_str, task):
        super().__init__(state_mapper, config_str, task)

    def preprocess(self, rl_agent, state: np.ndarray, action_mapper,
                   current_action_name: str, deploy: bool) -> np.ndarray:
        return state

    def should_allow_action(self, state: np.ndarray, action_name: str) -> bool:
        return False
