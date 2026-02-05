"""
Behavioral cloning dataset from an induced MDP with permissive ensemble policy,
using the OPTIMAL SCHEDULER from the induced MDP for labeling.

Loads a previously trained ensemble agent, builds the induced MDP using a
permissive policy (allowing any action that any ensemble member would take),
then extracts the optimal scheduler from this induced MDP and uses those
optimal actions as labels.

This differs from induced_ensemble_mdp_dataset.py which uses majority voting
for labels. This approach finds the optimal policy WITHIN the restricted
action space defined by the ensemble.
"""
import os
import json
import numpy as np
import stormpy
from mlflow.tracking import MlflowClient
from common.behavioral_cloning_dataset.dataset import BehavioralCloningDataset
from common.agents.agent_builder import AgentBuilder
from common.preprocessors.permissive_ensemble import PermissiveEnsemble


class InducedEnsembleOptimalDataset(BehavioralCloningDataset):
    """
    Dataset that extracts state-action pairs from an induced MDP with permissive
    ensemble, using the OPTIMAL SCHEDULER for labels.

    The permissive ensemble policy explores all states reachable by ANY ensemble
    member's action, and the optimal scheduler from model checking provides the
    labels (actions that maximize/minimize the objective).

    Config format: induced_ensemble_optimal;project_name;run_id;property;constant_definitions

    The property determines whether we maximize (Pmax) or minimize (Pmin) the objective.
    """

    def __init__(self, config):
        # Parse config: induced_ensemble_optimal;project_name;run_id;property;constant_definitions
        parts = config.split(";")
        self.name = parts[0]
        self.project_name = parts[1]
        self.run_id = parts[2]
        self.prop = parts[3]
        self.constant_definitions = parts[4] if len(parts) > 4 else ""

        self.X = None
        self.y = None
        self.loaded_agent = None  # Store the loaded ensemble agent for weight transfer

    def create(self, env):
        """
        Load ensemble agent, build induced MDP with permissive policy,
        extract optimal scheduler, and collect state-action pairs.
        """
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

        # Store the loaded agent for weight transfer
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

        # Diagnostic: count states with nondeterminism (multiple choices)
        multi_choice_states = sum(1 for s in model.states if len(s.actions) > 1)
        print(f"\nInduced MDP size: {model.nr_states} states, {model.nr_transitions} transitions")
        print(f"States with nondeterminism (multiple actions): {multi_choice_states}")
        print(f"Optimal value from induced MDP: {result}")

        # Extract state-action pairs from the optimal scheduler
        self._extract_state_action_pairs(model, scheduler, env)

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
            """Parse state valuation to numpy array using the same method as model_checker.
            This ensures identical state representation between permissive MDP building
            and model checking of the majority vote policy."""
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
        """
        states = []
        actions = []
        skipped_count = 0

        example_keys_ordered = list(json.loads(env.storm_bridge.state_json_example).keys())

        for state in model.states:
            # Get the optimal action from scheduler
            choice = scheduler.get_choice(state)
            chosen_action_index = choice.get_deterministic_choice()

            # Get the choice index and action labels
            choice_index = model.get_choice_index(state, chosen_action_index)
            action_labels = model.choice_labeling.get_labels_of_choice(choice_index)
            action_name = list(action_labels)[0] if action_labels else f"action_{chosen_action_index}"

            # Skip deadlock/unlabeled states
            if action_name == f"action_{chosen_action_index}":
                skipped_count += 1
                continue

            # Map action name to action index
            action_idx = env.action_mapper.action_name_to_action_index(action_name)
            if action_idx is None:
                skipped_count += 1
                continue

            # Parse state valuation to numpy array (same method as model_checker)
            state_valuation = model.state_valuations.get_json(state.id)
            raw_state_dict = json.loads(str(state_valuation))
            state_dict = {k: raw_state_dict[k] for k in example_keys_ordered if k in raw_state_dict}
            state_np = env.storm_bridge.parse_state(json.dumps(state_dict))

            states.append(state_np)
            actions.append(action_idx)

        # Convert to numpy arrays
        self.X = np.array(states)
        self.y = np.array(actions)

        print(f"\nCollected {len(self.X)} state-action pairs from optimal scheduler "
              f"({skipped_count} terminal states skipped)")

    def get_data(self):
        return {"X_train": self.X, "y_train": self.y, "X_test": None, "y_test": None}

    def initialize_agent_weights(self, target_agent):
        """
        Initialize target agent's weights from the loaded ensemble agent.

        This allows retraining to continue from existing weights rather than
        starting from scratch.

        Args:
            target_agent: The agent to initialize with weights from loaded ensemble
        """
        if self.loaded_agent is None:
            print("WARNING: No loaded agent available for weight transfer")
            return

        # Check if both agents are of the same type
        loaded_type = type(self.loaded_agent).__name__
        target_type = type(target_agent).__name__

        if loaded_type != target_type:
            print(f"WARNING: Agent type mismatch ({loaded_type} vs {target_type}), skipping weight transfer")
            return

        # Transfer weights based on agent type
        if hasattr(self.loaded_agent, 'networks') and hasattr(self.loaded_agent, 'trees') \
                and hasattr(target_agent, 'networks') and hasattr(target_agent, 'trees'):
            # Mixed Ensemble: transfer NN weights, train DTs fresh
            if len(self.loaded_agent.networks) == len(target_agent.networks):
                for i, (src_net, tgt_net) in enumerate(zip(self.loaded_agent.networks, target_agent.networks)):
                    tgt_net.load_state_dict(src_net.state_dict())
                print(f"Transferred weights from {len(target_agent.networks)} neural networks")
            else:
                print(f"WARNING: NN count mismatch ({len(self.loaded_agent.networks)} vs {len(target_agent.networks)})")
            print(f"Skipping tree transfer for {len(target_agent.trees)} decision trees (will train fresh)")
        elif hasattr(self.loaded_agent, 'networks') and hasattr(target_agent, 'networks'):
            # NN Ensemble: copy network weights
            if len(self.loaded_agent.networks) == len(target_agent.networks):
                for i, (src_net, tgt_net) in enumerate(zip(self.loaded_agent.networks, target_agent.networks)):
                    tgt_net.load_state_dict(src_net.state_dict())
                target_agent.is_trained = self.loaded_agent.is_trained
                print(f"Transferred weights from {len(target_agent.networks)} neural networks")
            else:
                print(f"WARNING: Network count mismatch ({len(self.loaded_agent.networks)} vs {len(target_agent.networks)})")
        elif hasattr(self.loaded_agent, 'trees') and hasattr(target_agent, 'trees'):
            # Decision Tree Ensemble: DO NOT transfer trees
            # We want to keep the target agent's diverse hyperparameters
            # Decision trees are fast to train, so no benefit from transfer
            print(f"Skipping tree transfer to preserve diverse hyperparameters in target agent")
        else:
            print("WARNING: Could not determine how to transfer weights for this agent type")
