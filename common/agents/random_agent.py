import numpy as np
from common.agents.black_box_agent import BlackBoxAgent


class RandomAgent(BlackBoxAgent):
    """Dummy black-box policy that picks actions uniformly at random."""

    def __init__(self, num_actions: int, seed: int = 0):
        super().__init__()
        self.num_actions = int(num_actions)
        self._rng = np.random.default_rng(seed)

    def sample_action(self, state: np.ndarray,
                      available_actions=None) -> int:
        return int(self._rng.integers(0, self.num_actions))
