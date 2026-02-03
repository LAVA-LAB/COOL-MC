from common.behavioral_cloning_dataset.dataset import *
import math


class AllOptimalDataset(BehavioralCloningDataset):
    """
    Behavioral cloning dataset that extracts ALL optimal state-action pairs.

    Unlike RawDataset which only extracts one optimal action per state,
    this dataset includes all actions that are equally optimal at each state.
    This is useful when multiple actions lead to the same optimal outcome.
    """

    def create(self, env):
        """
        Create the dataset by extracting all optimal state-action pairs.

        For each state, this method:
        1. Computes the optimal value for that state
        2. Evaluates all available actions
        3. Includes all actions whose value equals the optimal value

        Args:
            env: The environment to collect behavioral cloning data from.
        """
        prism_program = stormpy.parse_prism_program(self.prism_file)

        # Preprocess FIRST, then build suggestions (consistent with storm_bridge.py)
        prism_program = stormpy.preprocess_symbolic_input(
            prism_program, [], self.constant_definitions)[0].as_prism_program()

        # Label unlabelled commands AFTER preprocessing
        suggestions = dict()
        i = 0
        for module in prism_program.modules:
            for command in module.commands:
                if not command.is_labeled:
                    suggestions[command.global_index] = "tau_" + str(i)
                    i += 1

        prism_program = prism_program.label_unlabelled_commands(suggestions)

        # Parse property
        formulas = stormpy.parse_properties_for_prism_program(self.prop, prism_program)

        if not formulas or len(formulas) == 0:
            raise ValueError(
                f"Failed to parse property: '{self.prop}'. "
                "Please ensure the property is valid PCTL formula."
            )

        # Build model with state valuations and choice labels
        options = stormpy.BuilderOptions([f.raw_formula for f in formulas])
        options.set_build_state_valuations()
        options.set_build_choice_labels()
        model = stormpy.build_sparse_model_with_options(prism_program, options)

        # Perform model checking to get optimal values for all states
        result = stormpy.model_checking(model, formulas[0])

        # Extract all optimal state-action pairs
        states = []
        actions = []

        # Get state mapper for consistent state representation
        state_mapper = env.storm_bridge.state_mapper

        # Get example keys ordered from state_json_example to maintain consistency
        example_keys_ordered = list(json.loads(env.storm_bridge.state_json_example).keys())

        # Check if this is a reward property by examining the formula type
        formula = formulas[0].raw_formula
        is_reward_property = formula.is_reward_operator

        # Only use rewards if this is actually a reward property
        reward_model = None
        if is_reward_property and len(model.reward_models) > 0:
            reward_model = model.reward_models.get("", None)

        for state in model.states:
            # Get the optimal value for this state
            optimal_value = result.at(state.id)

            # Skip states with infinite optimal value
            if not math.isfinite(optimal_value):
                continue

            # Iterate through all available actions for this state
            for action_idx in range(model.get_nr_available_actions(state.id)):
                # Get the choice index for this action
                choice_index = model.get_choice_index(state, action_idx)

                # Compute the expected value of taking this action
                # For probability properties: sum(probability * next_state_value)
                # For reward properties: immediate_reward + sum(probability * next_state_value)
                action_value = 0.0

                # Add immediate reward ONLY if this is a reward property
                if is_reward_property and reward_model and reward_model.has_state_action_rewards:
                    action_value += reward_model.get_state_action_reward(choice_index)

                # Add expected future value
                for entry in model.transition_matrix.get_row(choice_index):
                    next_state = entry.column
                    probability = entry.value()
                    next_value = result.at(next_state)
                    # Break if next state has infinite value
                    if not math.isfinite(next_value):
                        action_value = float('inf')
                        break
                    action_value += probability * next_value

                # Skip infinite action values
                if not math.isfinite(action_value):
                    continue

                # Check if this action is optimal
                # Use small epsilon for floating point comparison
                is_optimal = abs(action_value - optimal_value) < 1e-6

                if is_optimal:
                    # Get action labels
                    action_labels = model.choice_labeling.get_labels_of_choice(choice_index)
                    action_name = list(action_labels)[0] if action_labels else f"action_{action_idx}"

                    # Skip deadlock states with dummy action
                    if action_name == f"action_{action_idx}":
                        continue

                    # Map action name to action index
                    # Skip this state-action pair if the action is not in the action_mapper
                    env_action_idx = env.action_mapper.action_name_to_action_index(action_name)
                    if env_action_idx is None:
                        continue

                    # Get state valuation in JSON format
                    state_valuation = model.state_valuations.get_json(state.id)
                    raw_state_dict = json.loads(str(state_valuation))

                    # Filter and reorder state dict to match state_json_example order
                    # This ensures consistency with parse_state which iterates JSON keys in order
                    state_dict = {k: raw_state_dict[k] for k in example_keys_ordered if k in raw_state_dict}

                    # Parse state to numpy array EXACTLY matching StormBridge.parse_state
                    arr = []
                    for k in state_dict:
                        value = state_dict[k]
                        # Convert booleans to 0/1 first
                        if isinstance(value, bool):
                            value = 1 if value else 0
                        # Then convert everything to int
                        arr.append(int(value))
                    state_np = np.array(arr, dtype=np.int32)

                    # Map state using state_mapper (applies disabled features filtering)
                    state_np = state_mapper.map(state_np)

                    states.append(state_np)
                    actions.append(env_action_idx)

        # Convert to numpy arrays
        self.X = np.array(states)
        self.y = np.array(actions)

    def get_data(self):
        """
        Retrieve the behavioral cloning dataset with all optimal state-action pairs.

        Returns:
            A dictionary with keys 'X_train', 'y_train', 'X_test', 'y_test'.
            X_train contains states, y_train contains corresponding optimal actions.
            Test data is set to None.
        """
        return {"X_train": self.X, "y_train": self.y, "X_test": None, "y_test": None}
