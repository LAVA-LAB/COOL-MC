import numpy as np
from common.agents.agent import Agent


class SubsetDecisionTreeAgent(Agent):
    """Wrapper agent that uses a decision tree trained on a feature subset.

    During model checking, select_action receives the full state vector.
    This wrapper selects only the relevant feature columns before predicting.
    """

    def __init__(self, classifier, feature_indices, number_of_actions):
        self.classifier = classifier
        self.feature_indices = feature_indices
        self.number_of_actions = number_of_actions

    def select_action(self, state: np.ndarray, deploy: bool = False) -> int:
        subset = state[self.feature_indices].reshape(1, -1)
        return int(self.classifier.predict(subset)[0])
