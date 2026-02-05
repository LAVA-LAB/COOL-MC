"""
Behavioral cloning dataset from an induced MDP with permissive ensemble policy,
using the OPTIMAL SCHEDULER from the induced MDP for labeling.

This version ACCUMULATES state-action pairs across multiple refinement episodes.
It stores the accumulated dataset to a file and loads it on subsequent runs,
allowing decision tree ensembles to be retrained on the full history of collected
state-action pairs rather than just the newest data.

This is particularly useful for decision trees which cannot be warm-started like
neural networks - they need to be retrained from scratch on the full dataset.

Config format: induced_ensemble_optimal_accumulated;project_name;run_id;property;constant_definitions;accumulated_data_path
"""
import os
import json
import numpy as np
import stormpy
from mlflow.tracking import MlflowClient
from common.behavioral_cloning_dataset.dataset import BehavioralCloningDataset
from common.agents.agent_builder import AgentBuilder
from common.preprocessors.permissive_ensemble import PermissiveEnsemble


class InducedEnsembleOptimalAccumulatedDataset(BehavioralCloningDataset):
    """
    Dataset that extracts state-action pairs from an induced MDP with permissive
    ensemble, using the OPTIMAL SCHEDULER for labels.

    Key difference from InducedEnsembleOptimalDataset:
    - ACCUMULATES state-action pairs across multiple refinement episodes
    - Stores accumulated data to a file
    - Loads previous data on subsequent runs
    - Useful for decision trees that need to retrain on full history

    Config format: induced_ensemble_optimal_accumulated;project_name;run_id;property;constant_definitions;accumulated_data_path

    The property determines whether we maximize (Pmax) or minimize (Pmin) the objective.
    """

    def __init__(self, config):
        # Parse config: induced_ensemble_optimal_accumulated;project_name;run_id;property;constant_definitions;accumulated_data_path
        parts = config.split(";")
        self.name = parts[0]
        self.project_name = parts[1]
        self.run_id = parts[2]
        self.prop = parts[3]
        self.constant_definitions = parts[4] if len(parts) > 4 else ""
        self.accumulated_data_path = parts[5] if len(parts) > 5 else ""

        self.X = None
        self.y = None
        self.loaded_agent = None  # Store the loaded ensemble agent

    def create(self, env):
        """
        Load ensemble agent, build induced MDP with permissive policy,
        extract optimal scheduler, collect state-action pairs, and
        merge with previously accumulated data.
        """
        # Load previously accumulated data if it exists
        accumulated_states = []
        accumulated_actions = []

        if self.accumulated_data_path and os.path.exists(self.accumulated_data_path):
            print(f"\nLoading previously accumulated data from {self.accumulated_data_path}")
            loaded = np.load(self.accumulated_data_path)
            accumulated_states = list(loaded['states'])
            accumulated_actions = list(loaded['actions'])
            print(f"Loaded {len(accumulated_states)} previously accumulated state-action pairs")
        else:
            print(f"\nNo previously accumulated data found, starting fresh")

        # Use MLflow client to get artifact path without starting a new run
        client = MlflowClient()

        # Get experiment by name
        experiment = client.get_experiment_by_name(self.project_name)
        if experiment is None:
            raise ValueError(f"Experiment '{self.project_name}' not found")

        # Get the artifact path for the run
        run = client.get_run(self.run_id)
        artifact_uri = run.info.artifact_uri.replace("file://", "")

        # Load command line arguments from the run
        meta_path = os.path.join(artifact_uri, "meta", "command_line_arguments.json")
        if not os.path.exists(meta_path):
            raise ValueError(f"Command line arguments not found at {meta_path}")

        with open(meta_path) as f:
            saved_args = json.load(f)

        # Get model path
        model_path = os.path.join(artifact_uri, "model")

        # Build agent using saved arguments
        agent = AgentBuilder.build_agent(
            model_path,
            saved_args,
            env.observation_space,
            env.action_space,
            env.action_mapper.actions
        )
        agent.load_env(env)

        # Store the loaded agent
        self.loaded_agent = agent

        print(f"Loaded ensemble agent from {self.project_name}/{self.run_id}")
        print(f"Algorithm: {saved_args['algorithm']}")

        # Verify this is an ensemble agent
        try:
            test_state = np.zeros(env.observation_space.shape[0])
            agent.get_ensemble_actions(test_state)
            print("Agent supports ensemble actions - using permissive exploration")
        except NotImplementedError:
            raise ValueError("Agent does not support ensemble actions. Use a different dataset type.")
        except RuntimeError:
            # Agent not trained - this is expected at this point
            pass

        # Build induced MDP with permissive ensemble policy
        print(f"\nBuilding induced MDP with permissive ensemble policy...")
        print(f"Property: {self.prop}")
        print(f"Constants: {self.constant_definitions}")

        # Build the induced model with scheduler extraction
        model, scheduler, result = self._build_induced_model_with_scheduler(env, agent)

        print(f"\nInduced MDP size: {model.nr_states} states, {model.nr_transitions} transitions")
        print(f"Optimal value from induced MDP: {result}")

        # Extract new state-action pairs from the optimal scheduler
        new_states, new_actions = self._extract_state_action_pairs(model, scheduler, env)

        print(f"\nCollected {len(new_states)} new state-action pairs from optimal scheduler")

        # Merge with accumulated data, keeping only unique state-action pairs
        # Use a dictionary keyed by state tuple for deduplication
        unique_pairs = {}

        # Add previously accumulated pairs
        for state, action in zip(accumulated_states, accumulated_actions):
            state_key = tuple(state)
            unique_pairs[state_key] = action

        # Add new pairs (may override if same state with different action from optimal scheduler)
        new_overrides = 0
        new_additions = 0
        for state, action in zip(new_states, new_actions):
            state_key = tuple(state)
            if state_key in unique_pairs:
                if unique_pairs[state_key] != action:
                    new_overrides += 1
                    unique_pairs[state_key] = action  # Use newer optimal action
            else:
                new_additions += 1
                unique_pairs[state_key] = action

        print(f"New unique states added: {new_additions}")
        print(f"Existing states with updated actions: {new_overrides}")

        # Convert to arrays
        self.X = np.array([list(s) for s in unique_pairs.keys()])
        self.y = np.array(list(unique_pairs.values()))

        print(f"\nTotal accumulated unique state-action pairs: {len(self.X)}")

        # Save accumulated data
        if self.accumulated_data_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.accumulated_data_path), exist_ok=True)
            np.savez(self.accumulated_data_path, states=self.X, actions=self.y)
            print(f"Saved accumulated data to {self.accumulated_data_path}")

        # Print action distribution
        unique_actions, counts = np.unique(self.y, return_counts=True)
        print(f"\nAction distribution in accumulated data:")
        for action, count in zip(unique_actions, counts):
            print(f"  Action {action}: {count} samples ({100*count/len(self.y):.1f}%)")

    def _build_induced_model_with_scheduler(self, env, agent):
        """
        Build the induced MDP using permissive ensemble policy and extract optimal scheduler.

        Returns:
            Tuple of (model, scheduler, result_value)
        """
        prism_program = stormpy.parse_prism_program(env.storm_bridge.path)

        # Preprocess with constants
        prism_program = stormpy.preprocess_symbolic_input(
            prism_program, [], self.constant_definitions)[0].as_prism_program()

        # Label unlabelled commands
        suggestions = dict()
        i = 0
        for module in prism_program.modules:
            for command in module.commands:
                if not command.is_labeled:
                    suggestions[command.global_index] = "tau_" + str(i)
                    i += 1
        prism_program = prism_program.label_unlabelled_commands(suggestions)

        # Parse property
        properties = stormpy.parse_properties(self.prop, prism_program)
        if not properties or len(properties) == 0:
            raise ValueError(f"Failed to parse property: '{self.prop}'")

        # Build options
        options = stormpy.BuilderOptions([p.raw_formula for p in properties])
        options.set_build_state_valuations()
        options.set_build_choice_labels(True)

        # Create simulator for incremental building
        simulator = stormpy.simulator.create_simulator(prism_program)
        simulator.set_action_mode(stormpy.simulator.SimulatorActionMode.GLOBAL_NAMES)

        # Get example keys ordered for consistent state representation
        example_keys_ordered = list(json.loads(env.storm_bridge.state_json_example).keys())

        def _parse_state_valuation(state_valuation):
            """Parse state valuation to numpy array using the same method as model_checker."""
            state_val_json = json.loads(str(state_valuation.to_json()))
            state_dict = {k: state_val_json[k] for k in example_keys_ordered if k in state_val_json}
            return env.storm_bridge.parse_state(json.dumps(state_dict))

        def incremental_building(state_valuation, action_index):
            """Allow actions from any ensemble member (permissive policy)."""
            simulator.restart(state_valuation)
            available_actions = sorted(simulator.available_actions())
            current_action_name = prism_program.get_action_name(action_index)

            # Check if action is available
            if current_action_name not in available_actions:
                return False

            # Parse state valuation to numpy array (same method as model_checker)
            state_np = _parse_state_valuation(state_valuation)

            # Get all ensemble member actions (permissive policy)
            try:
                ensemble_actions = agent.get_ensemble_actions(state_np)

                # Map each member's action to an action name, replacing unavailable
                # actions with available_actions[0] (same fallback as model_checker).
                # This ensures the permissive set is a SUPERSET of the majority vote.
                ensemble_action_names = []
                for a in ensemble_actions:
                    name = env.action_mapper.action_index_to_action_name(a)
                    if name not in available_actions:
                        name = available_actions[0]
                    ensemble_action_names.append(name)

                valid_ensemble_actions = list(set(ensemble_action_names))

                # Allow if current action is in any ensemble member's choice
                return current_action_name in valid_ensemble_actions
            except (NotImplementedError, RuntimeError):
                # Fallback to majority vote if ensemble actions not available
                action_idx = agent.select_action(state_np, deploy=True)
                selected_action = env.action_mapper.action_index_to_action_name(action_idx)
                if selected_action not in available_actions:
                    selected_action = available_actions[0]
                return current_action_name == selected_action

        # Build the induced model
        constructor = stormpy.make_sparse_model_builder(
            prism_program, options,
            stormpy.StateValuationFunctionActionMaskDouble(incremental_building)
        )
        model = constructor.build()

        # Model check with scheduler extraction
        result = stormpy.model_checking(model, properties[0], extract_scheduler=True)

        if not result.has_scheduler:
            raise ValueError("Model checking did not produce a scheduler")

        scheduler = result.scheduler
        initial_state = model.initial_states[0]
        result_value = result.at(initial_state)

        return model, scheduler, result_value

    def _extract_state_action_pairs(self, model, scheduler, env):
        """
        Extract state-action pairs from the model using the optimal scheduler.

        Returns:
            Tuple of (states_list, actions_list)
        """
        states = []
        actions = []

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
            action_idx = env.action_mapper.action_name_to_action_index(action_name)
            if action_idx is None:
                continue

            # Parse state valuation to numpy array (same method as model_checker)
            state_valuation = model.state_valuations.get_json(state.id)
            raw_state_dict = json.loads(str(state_valuation))
            state_dict = {k: raw_state_dict[k] for k in example_keys_ordered if k in raw_state_dict}
            state_np = env.storm_bridge.parse_state(json.dumps(state_dict))

            states.append(state_np)
            actions.append(action_idx)

        return states, actions

    def get_data(self):
        return {"X_train": self.X, "y_train": self.y, "X_test": None, "y_test": None}

    def initialize_agent_weights(self, target_agent):
        """
        For decision trees, we don't transfer weights - we retrain from scratch.
        This method exists for API compatibility but does nothing for decision trees.
        """
        # Decision trees cannot be warm-started like neural networks
        # They will be retrained from scratch on the full accumulated dataset
        print("Note: Decision trees will be trained from scratch on accumulated dataset")
