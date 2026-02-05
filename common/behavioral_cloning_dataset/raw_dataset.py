from common.behavioral_cloning_dataset.dataset import *


class RawDataset(BehavioralCloningDataset):


    def create(self, env):
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
            action_name = list(action_labels)[0] if action_labels else f"action_{chosen_action_index}"

            # Skip deadlock states with dummy action
            if action_name == f"action_{chosen_action_index}":
                continue

            # Map action name to action index
            # Skip this state-action pair if the action is not in the action_mapper
            action_idx = env.action_mapper.action_name_to_action_index(action_name)
            if action_idx is None:
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
        

        # Convert to numpy arrays
        self.X = np.array(states)
        self.y = np.array(actions)

    def get_data(self):
        return {"X_train": self.X, "y_train": self.y, "X_test": None, "y_test": None}