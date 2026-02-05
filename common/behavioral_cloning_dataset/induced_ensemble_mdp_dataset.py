"""
Behavioral cloning dataset from an induced MDP with permissive ensemble policy.

Loads a previously trained ensemble agent, builds the induced MDP using a
permissive policy (allowing any action that any ensemble member would take),
and extracts state-action pairs where the action is the MAJORITY VOTE.

This allows exploring more states than a single deterministic policy while
maintaining consistent labels (one action per state via majority voting).
"""
import os
import json
import numpy as np
from mlflow.tracking import MlflowClient
from common.behavioral_cloning_dataset.dataset import BehavioralCloningDataset
from common.agents.agent_builder import AgentBuilder
from common.preprocessors.permissive_ensemble import PermissiveEnsemble


class InducedEnsembleMDPDataset(BehavioralCloningDataset):
    """
    Dataset that extracts state-action pairs from an induced MDP with permissive ensemble.

    The permissive ensemble policy explores all states reachable by ANY ensemble
    member's action, but the collected labels use the MAJORITY VOTE action.

    Config format: induced_ensemble_mdp;project_name;run_id;property;constant_definitions
    """

    def __init__(self, config):
        # Parse config: induced_ensemble_mdp;project_name;run_id;property;constant_definitions
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
        Load ensemble agent and extract state-action pairs from induced MDP.

        Uses permissive policy for exploration but majority vote for labels.
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
            print("WARNING: Agent does not support ensemble actions, falling back to standard induced dataset")
        except RuntimeError:
            # Agent not trained - this is expected at this point
            pass

        # Create permissive ensemble preprocessor
        permissive_preprocessor = PermissiveEnsemble(
            env.storm_bridge.state_mapper,
            "permissive_ensemble",
            "rl_model_checking"
        )

        # Prepare property - add min/max if not specified (required for MDP model checking)
        prop = self.prop
        if prop.find("max") == -1 and prop.find("min") == -1:
            operator_str = prop[:1]
            prop = operator_str + "min" + prop[1:]
            print(f"Prepared property: {prop}")

        # Build induced MDP with permissive ensemble policy
        # The preprocessor allows all ensemble member actions during exploration
        print(f"\nBuilding induced MDP with permissive ensemble policy...")
        mdp_result, info = env.storm_bridge.model_checker.induced_markov_chain(
            agent,
            [permissive_preprocessor],  # Use permissive preprocessor for exploration
            env,
            self.constant_definitions,
            prop,
            collect_label_and_states=True
        )

        print(f"Model checking result: {mdp_result}")
        print(f"Induced MDP size: {info['model_size']} states, {info['model_transitions']} transitions")

        # Extract collected states
        collected_states = info['collected_states']

        if len(collected_states) == 0:
            raise ValueError("No states collected from induced MDP")

        # For each collected state, get the MAJORITY VOTE action (not all ensemble actions)
        # This ensures we have consistent labels (one action per state)
        print(f"\nCollecting majority vote actions for {len(collected_states)} states...")

        unique_states = {}
        for state in collected_states:
            state_key = tuple(state)
            if state_key not in unique_states:
                # Get majority vote action for this state
                action = agent.select_action(np.array(state), deploy=True)
                unique_states[state_key] = action

        # Convert to arrays
        self.X = np.array([list(s) for s in unique_states.keys()])
        self.y = np.array(list(unique_states.values()))

        print(f"Collected {len(self.X)} unique state-action pairs for behavioral cloning")

        # Print action distribution
        unique_actions, counts = np.unique(self.y, return_counts=True)
        print(f"\nAction distribution in collected data:")
        for action, count in zip(unique_actions, counts):
            print(f"  Action {action}: {count} samples ({100*count/len(self.y):.1f}%)")

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
        if hasattr(self.loaded_agent, 'networks') and hasattr(target_agent, 'networks'):
            # NN Ensemble: copy network weights
            if len(self.loaded_agent.networks) == len(target_agent.networks):
                for i, (src_net, tgt_net) in enumerate(zip(self.loaded_agent.networks, target_agent.networks)):
                    tgt_net.load_state_dict(src_net.state_dict())
                target_agent.is_trained = self.loaded_agent.is_trained
                print(f"Transferred weights from {len(target_agent.networks)} neural networks")
            else:
                print(f"WARNING: Network count mismatch ({len(self.loaded_agent.networks)} vs {len(target_agent.networks)})")
        elif hasattr(self.loaded_agent, 'trees') and hasattr(target_agent, 'trees'):
            # Decision Tree Ensemble: copy trees
            if len(self.loaded_agent.trees) == len(target_agent.trees):
                target_agent.trees = [tree for tree in self.loaded_agent.trees]
                target_agent.is_trained = self.loaded_agent.is_trained
                print(f"Transferred {len(target_agent.trees)} decision trees")
            else:
                print(f"WARNING: Tree count mismatch ({len(self.loaded_agent.trees)} vs {len(target_agent.trees)})")
        else:
            print("WARNING: Could not determine how to transfer weights for this agent type")
