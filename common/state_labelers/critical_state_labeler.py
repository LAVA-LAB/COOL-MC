import json
import numpy as np
import torch
from common.state_labelers.state_labeler import StateLabeler


class CriticalStateLabeler(StateLabeler):
    """Labels states as critical or non-critical based on neural network output confidence.

    A state is considered critical if the difference between max and min Q-values
    is BELOW the threshold (indicating uncertainty - all actions have similar values).
    States where the gap is ABOVE the threshold are non-critical (agent is confident).

    Configuration format: "critical_state;threshold=10.0"

    Labels added:
        - "critical": States where (max_Q - min_Q) < threshold (uncertain)
        - "non_critical": States where (max_Q - min_Q) >= threshold (confident)
    """

    def __init__(self, config_str: str):
        """Initialize the critical state labeler.

        Args:
            config_str: Configuration string with threshold
                        Format: "critical_state;threshold=<float>"
        """
        self.threshold = 10.0  # Default threshold
        super().__init__(config_str)

    def parse_config(self, config_str: str) -> None:
        """Parse the configuration string for threshold.

        Args:
            config_str: Configuration string (e.g., "critical_state;threshold=10.0")
        """
        parts = config_str.split(";")
        for part in parts[1:]:  # Skip the labeler name
            if "=" in part:
                key, value = part.split("=")
                key = key.strip()
                value = value.strip()
                if key == "threshold":
                    self.threshold = float(value)

        print(f"CriticalStateLabeler initialized with threshold: {self.threshold}")

    def mark_state_after_preprocessing(self, state: np.ndarray, agent, state_json: str) -> None:
        """Collect neural network output for each state during incremental building.

        Args:
            state: The preprocessed state as numpy array
            agent: The RL agent
            state_json: The raw state JSON string for identification
        """
        # Get raw neural network outputs
        with torch.no_grad():
            raw_outputs = agent.q_eval.forward(state)
            if isinstance(raw_outputs, torch.Tensor):
                raw_outputs = raw_outputs.cpu().numpy()

        # Calculate gap between max and min Q-values
        max_output = float(np.max(raw_outputs))
        min_output = float(np.min(raw_outputs))
        gap = max_output - min_output

        # Critical if gap is small (uncertain), non-critical if gap is large (confident)
        is_critical = gap < self.threshold

        self.collected_data[state_json] = {
            'max_output': max_output,
            'min_output': min_output,
            'gap': gap,
            'is_critical': is_critical
        }

    def label_states(self, model, env, agent) -> None:
        """Add critical and non_critical labels to the Storm model.

        Args:
            model: The built stormpy model
            env: The SafeGym environment
            agent: The RL agent
        """
        # Add label definitions
        model.labeling.add_label('critical')
        model.labeling.add_label('non_critical')

        critical_count = 0
        non_critical_count = 0
        all_gaps = []

        # Iterate through all states in the model
        for state_id in range(model.nr_states):
            state_json = str(model.state_valuations.get_json(state_id))

            # Check if we have collected data for this state
            if state_json in self.collected_data:
                gap = self.collected_data[state_json]['gap']
                all_gaps.append(gap)
                if self.collected_data[state_json]['is_critical']:
                    model.labeling.add_label_to_state('critical', state_id)
                    critical_count += 1
                else:
                    model.labeling.add_label_to_state('non_critical', state_id)
                    non_critical_count += 1
            else:
                # State not seen during incremental building - compute on the fly
                state_dict = json.loads(state_json)
                example_json = json.loads(env.storm_bridge.state_json_example)
                filtered_state = {k: v for k, v in state_dict.items() if k in example_json}
                state = env.storm_bridge.parse_state(json.dumps(filtered_state))

                with torch.no_grad():
                    raw_outputs = agent.q_eval.forward(state)
                    if isinstance(raw_outputs, torch.Tensor):
                        raw_outputs = raw_outputs.cpu().numpy()

                max_output = float(np.max(raw_outputs))
                min_output = float(np.min(raw_outputs))
                gap = max_output - min_output
                is_critical = gap < self.threshold

                all_gaps.append(gap)
                if is_critical:
                    model.labeling.add_label_to_state('critical', state_id)
                    critical_count += 1
                else:
                    model.labeling.add_label_to_state('non_critical', state_id)
                    non_critical_count += 1

        # Print gap distribution for debugging
        if all_gaps:
            print(f"CriticalStateLabeler: Q-value gap range = [{min(all_gaps):.2f}, {max(all_gaps):.2f}], threshold = {self.threshold}")
        print(f"CriticalStateLabeler: {critical_count} critical states, {non_critical_count} non-critical states")

    def get_label_names(self) -> list:
        """Return the label names added by this labeler.

        Returns:
            List containing 'critical' and 'non_critical'
        """
        return ['critical', 'non_critical']
