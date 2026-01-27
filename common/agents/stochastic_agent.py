
import numpy as np
from common.agents.agent import Agent


class StochasticAgent(Agent):
    """
    Base class for stochastic agents that use action probability distributions.

    All agents that inherit from this class will be detected by the model checker
    and their action probabilities will be used to create an induced DTMC instead
    of an induced MDP.

    Subclasses MUST implement the action_probability_distribution method.
    """

    def __init__(self):
        super().__init__()

    def action_probability_distribution(self, state: np.ndarray) -> np.ndarray:
        """
        Get the action probability distribution of the agent for a given state.

        This method MUST be implemented by all subclasses to return the probability
        distribution over all actions for the given state.

        Args:
            state (np.ndarray): The state of the environment.

        Returns:
            np.ndarray: The action probability distribution. Array of shape (num_actions,)
                       where each element is P(action|state) and sum equals 1.0.

        Example:
            For 4 actions: np.array([0.1, 0.6, 0.2, 0.1])
        """
        raise NotImplementedError("Subclasses must implement action_probability_distribution")

def to_tuple(number_of_elements : int, values : int):
    '''
    Creates a tuple with
    :param number_of_elements [int]: number of elements
    :param values [int]: element values
    :return: tuple[int]
    '''
    n_tuple: List[int] = []
    for i in range(number_of_elements):
        n_tuple.append(values)
    return tuple(n_tuple)
