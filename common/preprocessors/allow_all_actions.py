import numpy as np
from common.preprocessors.preprocessor import Preprocessor


class AllowAllActions(Preprocessor):
    """Preprocessor that allows all actions at every state.

    When used during model checking, this causes the model checker to explore
    the full MDP (all actions at every state) rather than only the agent's
    selected action. Useful for analyses that need the complete state space,
    such as the action_overlap state labeler.

    Configuration: "allow_all_actions"
    """

    def __init__(self, state_mapper, config_str, task):
        super().__init__(state_mapper, config_str, task)

    def preprocess(self, rl_agent, state: np.ndarray, action_mapper,
                   current_action_name: str, deploy: bool) -> np.ndarray:
        return state

    def should_allow_action(self, state: np.ndarray, action_name: str) -> bool:
        return True
