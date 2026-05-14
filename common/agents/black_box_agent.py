import numpy as np
from common.agents.agent import Agent


class BlackBoxAgent(Agent):
    """Base class for black-box policies used by the policy-sampling
    transition updater.

    The updater only queries the agent via :meth:`sample_action`; the
    underlying action distribution is never read directly. Subclasses
    must implement :meth:`sample_action` to draw a single action index
    from the policy's distribution at the given state.
    """

    def __init__(self):
        super().__init__()

    def sample_action(self, state: np.ndarray,
                      available_actions=None) -> int:
        """Draw one action a ~ pi(.|state).

        Args:
            state: numpy state vector.
            available_actions: optional list of action names allowed in
                the current state (``Act(s)``). Subclasses that want to
                expose the per-state action set to the policy (e.g. in
                an LLM prompt) should consume this; others may ignore it.
        """
        raise NotImplementedError(
            "BlackBoxAgent subclasses must implement sample_action"
        )

    def select_action(self, state: np.ndarray, deploy: bool = False):
        return self.sample_action(state)
