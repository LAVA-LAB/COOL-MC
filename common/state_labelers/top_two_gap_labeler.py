import json
import numpy as np
import torch
from common.state_labelers.state_labeler import StateLabeler


class TopTwoGapLabeler(StateLabeler):
    """Labels states based on the gap between max and second-max Q-values.

    A state is "confident" if the difference between the highest and second-highest
    Q-value is ABOVE the threshold (agent clearly prefers one action).
    A state is "not_confident" if the gap is BELOW the threshold (agent is uncertain
    between top choices).

    Configuration format: "top_two_gap;threshold=5.0"

    Labels added:
        - "confident": States where (max_Q - second_max_Q) >= threshold
        - "not_confident": States where (max_Q - second_max_Q) < threshold
    """

    def __init__(self, config_str: str):
        """Initialize the top-two gap labeler.

        Args:
            config_str: Configuration string with threshold
                        Format: "top_two_gap;threshold=<float>"
        """
        self.threshold = 5.0  # Default threshold
        super().__init__(config_str)

    def parse_config(self, config_str: str) -> None:
        """Parse the configuration string for threshold.

        Args:
            config_str: Configuration string (e.g., "top_two_gap;threshold=5.0")
        """
        parts = config_str.split(";")
        for part in parts[1:]:  # Skip the labeler name
            if "=" in part:
                key, value = part.split("=")
                key = key.strip()
                value = value.strip()
                if key == "threshold":
                    self.threshold = float(value)

        print(f"TopTwoGapLabeler initialized with threshold: {self.threshold}")

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

        # Sort Q-values in descending order
        sorted_q = np.sort(raw_outputs.flatten())[::-1]

        max_q = sorted_q[0]
        second_max_q = sorted_q[1] if len(sorted_q) > 1 else sorted_q[0]
        gap = max_q - second_max_q

        # Confident if gap is large (one action clearly better)
        is_confident = gap >= self.threshold

        self.collected_data[state_json] = {
            'max_q': max_q,
            'second_max_q': second_max_q,
            'gap': gap,
            'is_confident': is_confident
        }

    def label_states(self, model, env, agent) -> None:
        """Add confident and not_confident labels to the Storm model.

        Args:
            model: The built stormpy model
            env: The SafeGym environment
            agent: The RL agent
        """
        # Add label definitions
        model.labeling.add_label('confident')
        model.labeling.add_label('not_confident')

        confident_count = 0
        not_confident_count = 0
        all_gaps = []

        # Iterate through all states in the model
        for state_id in range(model.nr_states):
            state_json = str(model.state_valuations.get_json(state_id))

            # Check if we have collected data for this state
            if state_json in self.collected_data:
                gap = self.collected_data[state_json]['gap']
                all_gaps.append(gap)
                if self.collected_data[state_json]['is_confident']:
                    model.labeling.add_label_to_state('confident', state_id)
                    confident_count += 1
                else:
                    model.labeling.add_label_to_state('not_confident', state_id)
                    not_confident_count += 1
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

                sorted_q = np.sort(raw_outputs.flatten())[::-1]
                max_q = sorted_q[0]
                second_max_q = sorted_q[1] if len(sorted_q) > 1 else sorted_q[0]
                gap = max_q - second_max_q
                is_confident = gap >= self.threshold

                all_gaps.append(gap)
                if is_confident:
                    model.labeling.add_label_to_state('confident', state_id)
                    confident_count += 1
                else:
                    model.labeling.add_label_to_state('not_confident', state_id)
                    not_confident_count += 1

        # Print gap distribution for debugging
        if all_gaps:
            print(f"TopTwoGapLabeler: top-two gap range = [{min(all_gaps):.2f}, {max(all_gaps):.2f}], threshold = {self.threshold}")
        print(f"TopTwoGapLabeler: {confident_count} confident states, {not_confident_count} not_confident states")

    def get_label_names(self) -> list:
        """Return the label names added by this labeler.

        Returns:
            List containing 'confident' and 'not_confident'
        """
        return ['confident', 'not_confident']
