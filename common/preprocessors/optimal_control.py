"""
Optimal Control Preprocessor.

This preprocessor uses a behavioral cloning dataset to allow only optimal actions
during rl_model_checking. It enables permissive policies where multiple optimal
actions per state are allowed (similar to policy_abstraction).

Configuration format:
    "optimal_control;dataset_type;prism_file;property;constant_definitions"

Example:
    "optimal_control;all_optimal_dataset;../prism_files/transporter.prism;Rmin=? [ F jobs_done=2 ];MAX_JOBS=2,MAX_FUEL=10"
"""
import numpy as np
from common.preprocessors.preprocessor import Preprocessor
from common.behavioral_cloning_dataset.behavioral_cloning_dataset_builder import BehavioralCloningDatasetBuilder


class OptimalControl(Preprocessor):
    """
    A preprocessor that allows only optimal actions from a behavioral cloning dataset.

    During rl_model_checking, this preprocessor intercepts each state-action query
    and determines if the action is optimal. If so, it returns a (possibly modified)
    state that causes the agent to select that action, effectively allowing multiple
    optimal actions per state (permissive policy).
    """

    def __init__(self, state_mapper, config_str, task):
        super().__init__(state_mapper, config_str, task)
        self.parse_config(config_str)

        # Dataset and optimal action lookup
        self.dataset = None
        self.optimal_actions = {}  # state_tuple -> set of optimal action names
        self.action_mapper = None
        self.env = None

        # Cache for modified states: (state_tuple, action_name) -> modified_state
        self.state_action_cache = {}

        # Statistics
        self.total_queries = 0
        self.optimal_found = 0
        self.search_hits = 0

    def parse_config(self, config_str: str) -> None:
        """
        Parse the configuration string.

        Format: optimal_control;dataset_type;prism_file;property;constant_definitions

        Args:
            config_str: Configuration string
        """
        parts = config_str.split(";")
        # Parts 1-4 form the behavioral cloning dataset config
        self.bc_config = ";".join(parts[1:])

    def build_optimal_actions(self, env):
        """
        Build the optimal action lookup from the behavioral cloning dataset.

        Args:
            env: The SafeGym environment
        """
        if self.dataset is not None:
            return  # Already built

        self.env = env
        self.action_mapper = env.action_mapper

        # Build dataset
        dataset = BehavioralCloningDatasetBuilder.build(self.bc_config)
        if dataset is None:
            print(f"OptimalControl: Warning - Could not build dataset from config: {self.bc_config}")
            return

        dataset.create(env)
        self.dataset = dataset

        # Build optimal action lookup (state_tuple -> set of action names)
        data = dataset.get_data()
        X_train = data['X_train']
        y_train = data['y_train']

        self.optimal_actions = {}
        for state, action_idx in zip(X_train, y_train):
            # Convert to tuple, preserving dtype (works for both int and float)
            state_tuple = tuple(state)
            action_name = self.action_mapper.action_index_to_action_name(int(action_idx))

            if state_tuple not in self.optimal_actions:
                self.optimal_actions[state_tuple] = set()
            self.optimal_actions[state_tuple].add(action_name)

        print(f"OptimalControl: Loaded {len(self.optimal_actions)} states with optimal actions")
        print(f"OptimalControl: Total state-action pairs: {len(X_train)}")

    def get_optimal_action_names(self, state: np.ndarray) -> set:
        """
        Get the set of optimal action names for a given state.

        Args:
            state: State as numpy array

        Returns:
            Set of optimal action names, or empty set if state not found
        """
        state_tuple = tuple(np.array(state))
        return self.optimal_actions.get(state_tuple, set())

    def is_action_optimal(self, state: np.ndarray, action_name: str) -> bool:
        """
        Check if an action is optimal for the given state.

        Args:
            state: State as numpy array
            action_name: Name of the action to check

        Returns:
            True if the action is optimal for this state
        """
        optimal_names = self.get_optimal_action_names(state)
        return action_name in optimal_names

    def find_state_for_action(self, rl_agent, original_state: np.ndarray,
                               action_mapper, target_action_name: str, deploy: bool) -> np.ndarray:
        """
        Find a (possibly modified) state where the agent selects the target action.

        This enables permissive policies by finding state modifications that cause
        the agent to select different optimal actions.

        Args:
            rl_agent: The RL agent
            original_state: Original state
            action_mapper: Action mapper
            target_action_name: The action we want the agent to select
            deploy: Whether in deployment mode

        Returns:
            A state where agent selects target_action_name, or original_state if not found
        """
        state_tuple = tuple(np.array(original_state))
        cache_key = (state_tuple, target_action_name)

        # Check cache first
        if cache_key in self.state_action_cache:
            self.search_hits += 1
            return self.state_action_cache[cache_key]

        # Check if agent already selects this action with original state
        agent_action_idx = rl_agent.select_action(original_state, deploy)
        agent_action_name = action_mapper.action_index_to_action_name(agent_action_idx)
        if agent_action_name == target_action_name:
            self.state_action_cache[cache_key] = original_state
            return original_state

        # Search for a modified state where agent selects target action
        # Try small perturbations to each feature
        for i in range(len(original_state)):
            for delta in [-1, 1, -2, 2]:
                modified_state = original_state.copy()
                modified_state[i] = max(0, modified_state[i] + delta)  # Keep non-negative

                agent_action_idx = rl_agent.select_action(modified_state, deploy)
                agent_action_name = action_mapper.action_index_to_action_name(agent_action_idx)

                if agent_action_name == target_action_name:
                    self.state_action_cache[cache_key] = modified_state
                    return modified_state

        # Try combinations of two feature modifications
        for i in range(len(original_state)):
            for j in range(i + 1, len(original_state)):
                for delta_i in [-1, 1]:
                    for delta_j in [-1, 1]:
                        modified_state = original_state.copy()
                        modified_state[i] = max(0, modified_state[i] + delta_i)
                        modified_state[j] = max(0, modified_state[j] + delta_j)

                        agent_action_idx = rl_agent.select_action(modified_state, deploy)
                        agent_action_name = action_mapper.action_index_to_action_name(agent_action_idx)

                        if agent_action_name == target_action_name:
                            self.state_action_cache[cache_key] = modified_state
                            return modified_state

        # Not found - cache original state and return it
        # This means this optimal action won't be allowed (agent doesn't select it)
        self.state_action_cache[cache_key] = original_state
        return original_state

    def should_allow_action(self, state: np.ndarray, action_name: str) -> bool:
        """
        Directly determine if an action should be allowed at this state.

        This method is called by model_checker to directly control which actions
        are allowed, bypassing the agent's decision. This is more reliable than
        trying to find state modifications that make the agent select specific actions.

        Args:
            state: Current state as numpy array
            action_name: The action being checked

        Returns:
            True if the action should be allowed (is optimal), False otherwise
        """
        if self.dataset is None:
            return True  # No dataset loaded, allow all actions

        self.total_queries += 1

        # Check if action is optimal for this state
        is_optimal = self.is_action_optimal(state, action_name)
        if is_optimal:
            self.optimal_found += 1

        return is_optimal

    def preprocess(self, rl_agent, state: np.ndarray, action_mapper,
                   current_action_name: str, deploy: bool) -> np.ndarray:
        """
        Preprocess the state to allow optimal actions.

        Note: This method is kept for compatibility, but the primary mechanism
        for controlling actions is now through should_allow_action(), which
        is called directly by model_checker.

        Args:
            rl_agent: The RL agent
            state: Current state as numpy array
            action_mapper: Action mapper for converting indices to names
            current_action_name: The action being checked during incremental building
            deploy: Whether in deployment mode

        Returns:
            The original state (unchanged)
        """
        # Lazy initialization: build dataset when environment is available
        if self.dataset is None and hasattr(rl_agent, 'env') and rl_agent.env is not None:
            self.build_optimal_actions(rl_agent.env)

        # Return original state - action control is handled by should_allow_action
        return state

    def get_statistics(self) -> dict:
        """
        Get preprocessor statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'total_queries': self.total_queries,
            'optimal_found': self.optimal_found,
            'search_hits': self.search_hits,
            'cache_size': len(self.state_action_cache),
            'dataset_states': len(self.optimal_actions)
        }

    def save(self):
        """Save the preprocessor (no-op)."""
        pass

    def load(self, root_folder: str):
        """Load the preprocessor from folder (no-op, dataset is built lazily)."""
        pass

    def init_with_env(self, env):
        """
        Initialize the preprocessor with the environment.

        This should be called after the preprocessor is created and before
        model checking begins. It builds the optimal action lookup from
        the behavioral cloning dataset.

        Args:
            env: The SafeGym environment
        """
        self.build_optimal_actions(env)
