"""
Behavioral cloning dataset from an induced DTMC.

Loads a previously trained agent, builds the induced DTMC via model checking,
and extracts state-action pairs from all reachable states.
"""
import os
import json
import numpy as np
from mlflow.tracking import MlflowClient
from common.behavioral_cloning_dataset.dataset import BehavioralCloningDataset
from common.agents.agent_builder import AgentBuilder


class InducedDataset(BehavioralCloningDataset):
    """
    Dataset that extracts state-action pairs from an induced DTMC.

    Config format: induced_dataset;project_name;run_id;property;constant_definitions
    """

    def __init__(self, config):
        # Parse config: induced_dataset;project_name;run_id;property;constant_definitions
        parts = config.split(";")
        self.name = parts[0]
        self.project_name = parts[1]
        self.run_id = parts[2]
        self.prop = parts[3]
        self.constant_definitions = parts[4] if len(parts) > 4 else ""

        self.X = None
        self.y = None

    def create(self, env):
        """
        Load agent from previous run and extract state-action pairs from induced DTMC.
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

        print(f"Loaded agent from {self.project_name}/{self.run_id}")
        print(f"Algorithm: {saved_args['algorithm']}")

        # Prepare property - add min/max if not specified (required for MDP model checking)
        prop = self.prop
        if prop.find("max") == -1 and prop.find("min") == -1:
            operator_str = prop[:1]
            prop = operator_str + "min" + prop[1:]
            print(f"Prepared property: {prop}")

        # Build induced DTMC and collect state-action pairs
        mdp_result, info = env.storm_bridge.model_checker.induced_markov_chain(
            agent,
            None,  # No preprocessors for now
            env,
            self.constant_definitions,
            prop,
            collect_label_and_states=True
        )

        print(f"Model checking result: {mdp_result}")
        print(f"Induced DTMC size: {info['model_size']} states, {info['model_transitions']} transitions")

        # Extract collected state-action pairs
        collected_states = info['collected_states']
        collected_actions = info['collected_action_idizes']

        if len(collected_states) == 0:
            raise ValueError("No states collected from induced DTMC")

        self.X = np.array(collected_states)
        self.y = np.array(collected_actions)

        print(f"Collected {len(self.X)} state-action pairs for behavioral cloning")

    def get_data(self):
        return {"X_train": self.X, "y_train": self.y, "X_test": None, "y_test": None}
