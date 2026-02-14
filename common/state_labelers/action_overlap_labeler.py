import json
import numpy as np
from collections import defaultdict
from common.state_labelers.state_labeler import StateLabeler
from common.behavioral_cloning_dataset.all_optimal_dataset import AllOptimalDataset


class ActionOverlapLabeler(StateLabeler):
    """Labels states based on overlap between optimal actions of two properties.

    For each state, builds the set of optimal actions for two properties
    (e.g., Pmax survival and Pmax death). States where the intersection is
    non-empty are labeled "action_overlap" — meaning the best action for one
    objective is also optimal for the competing objective.

    Configuration format:
        "action_overlap;Pmax=? [ F \"survival\" ];Pmax=? [ F \"death\" ]"

    Labels added:
        - "action_overlap": States where optimal actions for both properties overlap
        - "no_action_overlap": States where optimal actions are disjoint
    """

    def __init__(self, config_str: str):
        self.prop1 = ""
        self.prop2 = ""
        super().__init__(config_str)

    def parse_config(self, config_str: str) -> None:
        parts = config_str.split(";")
        self.prop1 = parts[1].strip()
        self.prop2 = parts[2].strip()
        print(f"ActionOverlapLabeler initialized:")
        print(f"  Property 1: {self.prop1}")
        print(f"  Property 2: {self.prop2}")

    def mark_state_before_preprocessing(self, state: np.ndarray, agent, state_json: str) -> None:
        """Collect state_json -> state_tuple mapping during incremental building.

        Args:
            state: The raw state as numpy array (before preprocessing)
            agent: The RL agent
            state_json: The raw state JSON string for identification
        """
        if state_json not in self.collected_data:
            self.collected_data[state_json] = tuple(state)

    def _build_optimal_actions_map(self, env, prop):
        """Build mapping: state_tuple -> set of optimal action indices."""
        prism_path = env.storm_bridge.path
        constant_defs = env.storm_bridge.constant_definitions

        config = f"all_optimal_dataset;{prism_path};{prop};{constant_defs}"
        dataset = AllOptimalDataset(config)
        dataset.create(env)
        data = dataset.get_data()

        X, y = data['X_train'], data['y_train']

        state_to_actions = defaultdict(set)
        for i in range(len(X)):
            state_key = tuple(X[i])
            state_to_actions[state_key].add(y[i])

        return state_to_actions

    def label_states(self, model, env, agent) -> None:
        """Add overlap labels to the Storm model.

        Builds optimal action maps for both properties, then labels each
        state based on whether the optimal action sets overlap.

        Args:
            model: The built stormpy model
            env: The SafeGym environment
            agent: The RL agent
        """
        # Build optimal action maps for both properties
        print(f"\nBuilding optimal action map for: {self.prop1}")
        map1 = self._build_optimal_actions_map(env, self.prop1)
        print(f"  States with optimal actions: {len(map1)}")

        print(f"Building optimal action map for: {self.prop2}")
        map2 = self._build_optimal_actions_map(env, self.prop2)
        print(f"  States with optimal actions: {len(map2)}")

        # Add label definitions
        model.labeling.add_label('action_overlap')
        model.labeling.add_label('no_action_overlap')

        action_labels = env.action_mapper.actions
        overlap_count = 0
        no_overlap_count = 0
        not_found_count = 0
        overlap_action_counts = defaultdict(int)

        for state_id in range(model.nr_states):
            state_json = str(model.state_valuations.get_json(state_id))

            # Look up the state tuple collected during incremental building
            state_key = self.collected_data.get(state_json)

            if state_key is None:
                # State not seen during incremental building — compute on the fly
                state_dict = json.loads(state_json)
                example_json = json.loads(env.storm_bridge.state_json_example)
                filtered_state = {k: v for k, v in state_dict.items()
                                  if k in example_json}
                state_np = env.storm_bridge.parse_state(json.dumps(filtered_state))
                state_key = tuple(state_np)

            actions1 = map1.get(state_key, set())
            actions2 = map2.get(state_key, set())

            if not actions1 and not actions2:
                # State not in either dataset (e.g., absorbing/terminal state)
                model.labeling.add_label_to_state('no_action_overlap', state_id)
                not_found_count += 1
                continue

            overlap = actions1 & actions2
            if overlap:
                model.labeling.add_label_to_state('action_overlap', state_id)
                overlap_count += 1
                for a in overlap:
                    label = action_labels[a] if a < len(action_labels) else str(a)
                    overlap_action_counts[label] += 1
            else:
                model.labeling.add_label_to_state('no_action_overlap', state_id)
                no_overlap_count += 1

        # Summary
        total = model.nr_states
        print(f"\n{'='*80}")
        print("Action Overlap Labeler Results")
        print(f"{'='*80}")
        print(f"Total states:        {total}")
        print(f"Action overlap:      {overlap_count} ({overlap_count/total*100:.1f}%)")
        print(f"No action overlap:   {no_overlap_count} ({no_overlap_count/total*100:.1f}%)")
        if not_found_count > 0:
            print(f"Not in either set:   {not_found_count} ({not_found_count/total*100:.1f}%)")

        if overlap_action_counts:
            print(f"\nOverlapping actions (count of states where action overlaps):")
            for action, count in sorted(overlap_action_counts.items(),
                                        key=lambda x: -x[1]):
                print(f"  {action}: {count} states")

    def get_label_names(self) -> list:
        return ['action_overlap', 'no_action_overlap']
