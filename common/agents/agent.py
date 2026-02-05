
import numpy as np
class Agent():

    def __init__(self):
        pass

    def select_action(self, state : np.ndarray, deploy : bool =False):
        """
        The agent gets the OpenAI Gym state and makes a decision.

        Args:
            state (np.ndarray): The state of the OpenAI Gym.
            deploy (bool, optional): If True, do not add randomness to the decision making (e.g. deploy=True in model checking). Defaults to False.
        """
        pass

    def store_experience(self, state  : np.ndarray, action : int, reward : float, next_state : np.ndarray, terminal : bool):
        """
        Stores RL agent training experience.

        Args:
            state (np.ndarray): State
            action (int): Chosen Action
            reward (float): Received Reward
            next_state (np.ndarray): Next State
            terminal (bool): Episode ended?
        """
        pass

    def step_learn(self):
        """
        This method is called every step in the training environment.
        In this method, the agent learns.
        """
        pass

    def episodic_learn(self):
        """
        This method is called in the end of every training episode.
        In this method, the agent learns.
        """
        pass

    def raw_outputs(self, state: np.ndarray) -> np.ndarray:
        """
        Get the raw outputs of the agent for a given state.

        Args:
            state (np.ndarray): The state of the environment.
        Returns:
            np.ndarray: The raw outputs of the agent.
        """
        pass

    def get_ensemble_actions(self, state: np.ndarray) -> list[int]:
        """
        Get individual actions from all ensemble members for a given state.

        For ensemble agents (e.g., Random Forest, Decision Tree Ensemble),
        this returns the action selected by each individual policy/tree.

        Args:
            state (np.ndarray): The state of the environment.

        Returns:
            list[int]: List of actions from each ensemble member.

        Raises:
            NotImplementedError: If the agent is not an ensemble agent.
        """
        raise NotImplementedError("This agent does not support ensemble actions.")

    def model_checking_learn(self, model_checking_result, model_checking_info, model_checker=None):
        """
        This method is called in after model checking was executed and gets the model checking result passed.
        """
        pass

    def behavioral_cloning(self, env, data: dict, epochs: int = 100, accuracy_threshold: float = 100.0) -> tuple[int | None, float | None, float | None, float | None, float | None]:
        """
        Perform supervised training on a behavioral cloning dataset.

        Args:
            data: The behavioral cloning dataset.
            epochs: Number of training epochs.
            accuracy_threshold: Stop training early when accuracy reaches this threshold (in percent).

        Returns:
            A tuple of (training_epoch, train_accuracy, test_accuracy, train_loss, test_loss).
            Values may be None if not available.
        """
        return None, None, None, None, None

    def get_hyperparameters(self):
        """
        Get the RL agent hyperparameters
        """
        pass

    def save(self):
        """
        Saves the RL agent in the MLFlow experiment.
        """
        pass

    def load(self, root_folder:str):
        """
        Loads the RL agent from the folder

        Args:
            root_folder ([str]): Path the the agent folder
        """
        pass

    def load_env(self, env):
        """
        Further loading, when the environment is loaded.
        """
        pass

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
