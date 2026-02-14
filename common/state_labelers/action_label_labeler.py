import json
import numpy as np
from common.state_labelers.state_labeler import StateLabeler


class ActionLabelLabeler(StateLabeler):
    """Labels each state with the action name chosen by the trained policy.

    For each state in the induced model, this labeler queries the agent's
    policy (via select_action) and adds a label matching the chosen action name.
    This allows writing PCTL properties that reason about action sequences,
    e.g., P=? [ "treat_vasopressors" U "survival" ] to check the probability
    of reaching survival while the policy keeps choosing vasopressors.

    Configuration format: "action_label"

    Labels added:
        One label per unique action name discovered during building
        (e.g., "f0_v0", "f2_v4", "tau_0", etc.)
    """

    def __init__(self, config_str: str):
        self.state_actions = {}  # state_json -> action_index
        super().__init__(config_str)

    def parse_config(self, config_str: str) -> None:
        print("ActionLabelLabeler initialized")

    def mark_state_after_preprocessing(self, state: np.ndarray, agent, state_json: str) -> None:
        """Record the policy's chosen action for each state during incremental building."""
        action_index = agent.select_action(state, True)
        self.state_actions[state_json] = action_index

    def label_states(self, model, env, agent) -> None:
        """Add per-action labels to the Storm model."""
        action_names = env.action_mapper.actions
        action_counts = {}

        # Add a label for each possible action
        for action_name in action_names:
            if action_name not in model.labeling.get_labels():
                model.labeling.add_label(action_name)
            action_counts[action_name] = 0

        # Label each state with its policy-chosen action
        for state_id in range(model.nr_states):
            state_json = str(model.state_valuations.get_json(state_id))

            if state_json in self.state_actions:
                action_index = self.state_actions[state_json]
            else:
                # State not seen during incremental building - compute on the fly
                state_dict = json.loads(state_json)
                example_json = json.loads(env.storm_bridge.state_json_example)
                filtered_state = {k: v for k, v in state_dict.items() if k in example_json}
                state = env.storm_bridge.parse_state(json.dumps(filtered_state))
                action_index = agent.select_action(state, True)

            # Map index to action name
            if action_index < len(action_names):
                action_name = action_names[action_index]
            else:
                action_name = action_names[0]

            model.labeling.add_label_to_state(action_name, state_id)
            action_counts[action_name] = action_counts.get(action_name, 0) + 1

        # Print summary
        print(f"\nActionLabelLabeler: Labeled {model.nr_states} states")
        for name, count in sorted(action_counts.items()):
            pct = count / model.nr_states * 100 if model.nr_states > 0 else 0
            print(f"  {name}: {count} states ({pct:.1f}%)")

    def has_dynamic_labels(self) -> bool:
        return True
