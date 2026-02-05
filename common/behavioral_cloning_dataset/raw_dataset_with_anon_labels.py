from common.behavioral_cloning_dataset.dataset import *


class RawDatasetWithAnonLabels(BehavioralCloningDataset):
    """
    Behavioral cloning dataset that includes ALL states, including those with
    unlabeled/anonymous actions (tau_X).

    This dataset properly labels unlabeled commands AFTER preprocessing to ensure
    consistency with the action_mapper. This is important for models like zeroconf
    where goal states have unlabeled self-loops like `[] l=4 -> true;`
    """

    def create(self, env):
        prism_program = stormpy.parse_prism_program(self.prism_file)

        # Preprocess with constant definitions FIRST (like action_mapper does)
        prism_program = stormpy.preprocess_symbolic_input(
            prism_program, [], self.constant_definitions)[0].as_prism_program()

        # Label unlabelled commands AFTER preprocessing (consistent with action_mapper)
        suggestions = dict()
        tau_counter = 0
        for module in prism_program.modules:
            for command in module.commands:
                if not command.is_labeled:
                    suggestions[command.global_index] = f"tau_{tau_counter}"
                    tau_counter += 1

        prism_program = prism_program.label_unlabelled_commands(suggestions)

        # Parse property
        formulas = stormpy.parse_properties_for_prism_program(self.prop, prism_program)

        if not formulas or len(formulas) == 0:
            raise ValueError(
                f"Failed to parse property: '{self.prop}'. "
                "Please ensure the property is valid PCTL formula."
            )

        # Build model with state valuations
        options = stormpy.BuilderOptions([f.raw_formula for f in formulas])
        options.set_build_state_valuations()
        options.set_build_choice_labels()
        model = stormpy.build_sparse_model_with_options(prism_program, options)

        # Perform model checking and extract scheduler
        result = stormpy.model_checking(model, formulas[0], extract_scheduler=True)

        # Print the optimal value for verification
        initial_state = model.initial_states[0]
        optimal_value = result.at(initial_state)
        print(f"Optimal value from scheduler extraction: {optimal_value}")
        print(f"Total states in model: {model.nr_states}")

        if not result.has_scheduler:
            raise ValueError("Model checking did not produce a scheduler")

        scheduler = result.scheduler

        # Extract optimal state-action pairs
        states = []
        actions = []
        stats = {
            'labeled': 0,
            'tau_mapped': 0,
            'fallback_first_available': 0,
            'skipped': 0
        }

        # Get state mapper for consistent state representation
        state_mapper = env.storm_bridge.state_mapper

        # Get example keys ordered from state_json_example to maintain consistency
        example_keys_ordered = list(json.loads(env.storm_bridge.state_json_example).keys())

        for state in model.states:
            # Get the optimal action from scheduler
            choice = scheduler.get_choice(state)
            chosen_action_index = choice.get_deterministic_choice()

            # Get the choice index and action labels
            choice_index = model.get_choice_index(state, chosen_action_index)
            action_labels = model.choice_labeling.get_labels_of_choice(choice_index)
            action_name = list(action_labels)[0] if action_labels else None

            action_idx = None

            # Try to map the action by name
            if action_name:
                # First, try direct mapping (works for labeled actions and tau_X)
                action_idx = env.action_mapper.action_name_to_action_index(action_name)
                if action_idx is not None:
                    if action_name.startswith("tau_"):
                        stats['tau_mapped'] += 1
                    else:
                        stats['labeled'] += 1

            # If action_name was None or mapping failed, try alternatives
            if action_idx is None:
                # Check if this is an unlabeled action that should be tau_X
                # The chosen_action_index might correspond to a tau action
                tau_name = f"tau_{chosen_action_index}"
                action_idx = env.action_mapper.action_name_to_action_index(tau_name)
                if action_idx is not None:
                    stats['tau_mapped'] += 1

            # If still no match, use first available action as last resort
            # This handles absorbing states where action doesn't matter
            if action_idx is None:
                if env.action_mapper.get_action_count() > 0:
                    action_idx = 0
                    stats['fallback_first_available'] += 1
                else:
                    stats['skipped'] += 1
                    continue

            # Get state valuation in JSON format
            state_valuation = model.state_valuations.get_json(state.id)
            raw_state_dict = json.loads(str(state_valuation))

            # Filter and reorder state dict to match state_json_example order
            state_dict = {k: raw_state_dict[k] for k in example_keys_ordered if k in raw_state_dict}

            # Parse state to numpy array EXACTLY matching StormBridge.parse_state
            arr = []
            for k in state_dict:
                value = state_dict[k]
                if isinstance(value, bool):
                    value = 1 if value else 0
                arr.append(int(value))
            state_np = np.array(arr, dtype=np.int32)

            # Map state using state_mapper (applies disabled features filtering)
            state_np = state_mapper.map(state_np)

            # Check if compressed state representation is available
            if state_mapper.has_compressed_state_representation():
                try:
                    # Convert mapped state to string representation
                    state_str = state_mapper.state_to_str(state_np)
                    # Load decompressed state from file
                    state_np = state_mapper.decompress_state(state_str)
                except (FileNotFoundError, ValueError):
                    # If decompressed state not found, use mapped state
                    pass

            states.append(state_np)
            actions.append(action_idx)

        print(f"Dataset created: {len(states)} state-action pairs")
        print(f"  - Labeled actions: {stats['labeled']}")
        print(f"  - Tau actions mapped: {stats['tau_mapped']}")
        if stats['fallback_first_available'] > 0:
            print(f"  - Fallback to first action: {stats['fallback_first_available']}")
        if stats['skipped'] > 0:
            print(f"  - Skipped (no valid action): {stats['skipped']}")

        # Convert to numpy arrays
        self.X = np.array(states)
        self.y = np.array(actions)

    def get_data(self):
        return {"X_train": self.X, "y_train": self.y, "X_test": None, "y_test": None}
