import numpy as np


class StateLabeler:
    """Base class for state labelers.

    State labelers add custom labels to states in the Storm model after
    the incremental building process completes. Labels can be used in
    property specifications (e.g., P=? ["critical" U "goal"]).
    """

    def __init__(self, config_str: str):
        """Initialize the state labeler.

        Args:
            config_str: Configuration string (e.g., "critical_state;min=0.3;max=0.7")
        """
        self.config_str = config_str
        self.collected_data = {}  # Store data during incremental building
        self.available_actions = None  # Current state's available actions (set by collect_available_actions)
        self.parse_config(config_str)

    def parse_config(self, config_str: str) -> None:
        """Parse the configuration string.

        Args:
            config_str: Configuration string with semicolon-separated key=value pairs
        """
        pass

    def collect_available_actions(self, available_actions: list) -> None:
        """Store the available actions for the current state being processed.

        Called during incremental building before mark_state_before_preprocessing
        and mark_state_after_preprocessing. The available_actions attribute holds
        the sorted list of action name strings from simulator.available_actions()
        for the state currently being visited.

        Args:
            available_actions: Sorted list of available action name strings
        """
        self.available_actions = available_actions

    def mark_state_before_preprocessing(self, state: np.ndarray, agent, state_json: str) -> None:
        """Called during incremental building BEFORE preprocessing.

        Use this to collect information about the raw state before any
        transformations are applied. self.available_actions contains the
        actions available at this state.

        Args:
            state: The raw state as numpy array (before preprocessing)
            agent: The RL agent
            state_json: The raw state JSON string for identification
        """
        pass

    def mark_state_after_preprocessing(self, state: np.ndarray, agent, state_json: str) -> None:
        """Called during incremental building AFTER preprocessing.

        Use this to collect information about the preprocessed state.
        This is typically where you compute agent outputs for labeling.
        self.available_actions contains the actions available at this state.

        Args:
            state: The preprocessed state as numpy array
            agent: The RL agent
            state_json: The raw state JSON string for identification
        """
        pass

    def label_states(self, model, env, agent) -> None:
        """Add labels to the built Storm model.

        Called after model = constructor.build() completes.
        Use model.labeling.add_label() and model.labeling.add_label_to_state()
        to add custom labels.

        Args:
            model: The built stormpy model
            env: The SafeGym environment
            agent: The RL agent
        """
        pass

    def get_label_names(self) -> list:
        """Return list of label names this labeler adds.

        Returns:
            List of label name strings
        """
        return []

    def has_dynamic_labels(self) -> bool:
        """Whether this labeler adds labels only known at build time.

        Override to return True if the label names are not known at init
        (e.g., collected during incremental building). The model checker
        uses this to force context-free property parsing.

        Returns:
            True if labels are determined dynamically during building
        """
        return False
