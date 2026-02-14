import json
import csv
import numpy as np
from common.state_labelers.state_labeler import StateLabeler


class PermutationImportanceLabeler(StateLabeler):
    """Labels each state with the feature most important to the policy's decision.

    For each state, performs permutation feature importance: each specified
    feature is permuted (replaced with values from other states) and the
    fraction of permutations that change the policy's chosen action is measured.
    The state is labeled with the feature whose permutation most frequently
    changes the action — i.e., the feature the policy relies on most at that state.

    This enables PCTL queries about feature-driven decision regions, e.g.:
        P=? [ F "imp_GCS" ]          — reach a state where GCS drives the decision
        P=? [ "imp_SOFA" U "death" ] — die through states where SOFA drives the decision

    Configuration format:
        "permutation_importance;feature1,feature2,...[;n_permutations=N][;output_file.csv]"

    Parameters:
        features:        Comma-separated list of feature names to evaluate
        n_permutations:  Number of random permutations per feature per state (default: 20)
        output_file:     Optional CSV path for per-state importance scores

    Labels added:
        "imp_{feature_name}" for each specified feature
        Each state receives exactly one label: the most important feature.
        If all features have zero importance (action never changes), the state
        gets the label "imp_none".
    """

    def __init__(self, config_str: str):
        self.features_to_test = []
        self.n_permutations = 20
        self.output_file = ""
        self.after_preprocess_data = {}  # state_json -> preprocessed state
        super().__init__(config_str)

    def parse_config(self, config_str: str) -> None:
        parts = config_str.split(";")
        if len(parts) < 2 or not parts[1].strip():
            raise ValueError(
                "PermutationImportanceLabeler requires at least feature names. "
                "Format: permutation_importance;feat1,feat2[;n_permutations=N][;output.csv]")

        # Parse feature names (second part)
        self.features_to_test = [f.strip() for f in parts[1].split(",")]

        # Parse optional parameters
        for part in parts[2:]:
            part = part.strip()
            if not part:
                continue
            if part.startswith("n_permutations="):
                self.n_permutations = int(part.split("=")[1])
            elif part.endswith(".csv"):
                self.output_file = part
            else:
                print(f"Warning: Unrecognized parameter '{part}'")

        print(f"PermutationImportanceLabeler initialized:")
        print(f"  Features: {self.features_to_test}")
        print(f"  Permutations per feature: {self.n_permutations}")
        if self.output_file:
            print(f"  Output CSV: {self.output_file}")

    def mark_state_before_preprocessing(self, state: np.ndarray, agent, state_json: str) -> None:
        """Collect raw/decompressed state vectors during incremental building."""
        if state_json not in self.collected_data:
            self.collected_data[state_json] = state.copy()

    def mark_state_after_preprocessing(self, state: np.ndarray, agent, state_json: str) -> None:
        """Collect preprocessed state vectors during incremental building."""
        if state_json not in self.after_preprocess_data:
            self.after_preprocess_data[state_json] = state.copy()

    def label_states(self, model, env, agent) -> None:
        """Compute per-state permutation importance and label each state."""
        # Get feature names
        state_mapper = env.storm_bridge.state_mapper
        if state_mapper.has_compressed_state_representation():
            feature_names = state_mapper.get_compressed_feature_names()
        else:
            feature_names = state_mapper.get_feature_names()

        if feature_names is None:
            n_features = len(next(iter(self.collected_data.values())))
            feature_names = [f"f{i}" for i in range(n_features)]

        # Map specified feature names to indices
        feature_indices = {}
        for fname in self.features_to_test:
            if fname in feature_names:
                feature_indices[fname] = feature_names.index(fname)
            else:
                print(f"WARNING: Feature '{fname}' not found in feature names. "
                      f"Available: {feature_names}")

        if not feature_indices:
            print("PermutationImportanceLabeler: No valid features found.")
            return

        # Build state_id -> state vector mapping (use preprocessed if available, else raw)
        state_vectors = {}
        state_raw_vectors = {}
        for state_id in range(model.nr_states):
            state_json = str(model.state_valuations.get_json(state_id))
            if state_json in self.after_preprocess_data:
                state_vectors[state_id] = self.after_preprocess_data[state_json]
                state_raw_vectors[state_id] = self.collected_data.get(state_json)
            elif state_json in self.collected_data:
                state_vectors[state_id] = self.collected_data[state_json]
                state_raw_vectors[state_id] = self.collected_data[state_json]
            else:
                # Fallback: decompress on the fly
                try:
                    state_dict = json.loads(state_json)
                    example_json = json.loads(env.storm_bridge.state_json_example)
                    filtered = {k: v for k, v in state_dict.items()
                                if k in example_json}
                    state_vec = env.storm_bridge.parse_state(json.dumps(filtered))
                    state_vectors[state_id] = state_vec
                    state_raw_vectors[state_id] = state_vec
                except Exception:
                    state_vectors[state_id] = None
                    state_raw_vectors[state_id] = None

        # Collect valid state vectors into a matrix for sampling permutation values
        valid_ids = [sid for sid in state_vectors if state_vectors[sid] is not None]
        if not valid_ids:
            print("PermutationImportanceLabeler: No states with features found.")
            return

        all_vectors = np.array([state_vectors[sid] for sid in valid_ids])
        n_states = len(valid_ids)

        print(f"\nPermutation Importance Analysis:")
        print(f"  States: {n_states}")
        print(f"  Features: {list(feature_indices.keys())}")
        print(f"  Permutations per feature: {self.n_permutations}")

        # Add labels
        label_names = [f"imp_{fname}" for fname in feature_indices]
        label_names.append("imp_none")
        for label in label_names:
            if label not in model.labeling.get_labels():
                model.labeling.add_label(label)

        # Compute permutation importance for each state
        label_counts = {label: 0 for label in label_names}
        per_state_results = []  # for CSV output
        np.random.seed(42)  # reproducibility

        for i, state_id in enumerate(valid_ids):
            state_vec = state_vectors[state_id]
            original_action = agent.select_action(state_vec, True)

            importance_scores = {}
            for fname, fidx in feature_indices.items():
                action_changes = 0
                # Sample permutation values from the distribution of this feature
                perm_indices = np.random.randint(0, n_states, size=self.n_permutations)
                for perm_idx in perm_indices:
                    permuted = state_vec.copy()
                    permuted[fidx] = all_vectors[perm_idx, fidx]
                    permuted_action = agent.select_action(permuted, True)
                    if permuted_action != original_action:
                        action_changes += 1
                importance_scores[fname] = action_changes / self.n_permutations

            # Find most important feature
            max_importance = max(importance_scores.values())
            if max_importance > 0:
                most_important = max(importance_scores, key=importance_scores.get)
                label = f"imp_{most_important}"
            else:
                most_important = "none"
                label = "imp_none"

            model.labeling.add_label_to_state(label, state_id)
            label_counts[label] += 1

            # Store for CSV
            if self.output_file:
                row = {"state_id": state_id, "most_important": most_important}
                row.update({f"imp_{fname}": f"{score:.4f}"
                            for fname, score in importance_scores.items()})
                per_state_results.append(row)

            # Progress
            if (i + 1) % 500 == 0 or (i + 1) == n_states:
                print(f"  Processed {i + 1}/{n_states} states...")

        # Label remaining states (terminal/absorbing) with imp_none
        for state_id in range(model.nr_states):
            if state_id not in valid_ids:
                model.labeling.add_label_to_state("imp_none", state_id)
                label_counts["imp_none"] += 1

        # Print summary
        print(f"\nPermutation Importance Labels:")
        print(f"{'Label':<35s} {'Count':>8s} {'Percentage':>10s}")
        print("-" * 55)
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            pct = count / model.nr_states * 100 if model.nr_states > 0 else 0
            print(f"  {label:<33s} {count:>8d} {pct:>9.1f}%")

        # Compute aggregate importance scores across all states
        print(f"\nAggregate Feature Importance (mean action-change rate across states):")
        agg_importance = {fname: 0.0 for fname in feature_indices}
        for row in per_state_results if per_state_results else []:
            for fname in feature_indices:
                agg_importance[fname] += float(row[f"imp_{fname}"])
        if n_states > 0:
            for fname in agg_importance:
                agg_importance[fname] /= n_states
        ranked = sorted(agg_importance.items(), key=lambda x: x[1], reverse=True)
        for rank, (fname, score) in enumerate(ranked, 1):
            print(f"  {rank}. {fname}: {score:.4f}")

        # Save CSV
        if self.output_file and per_state_results:
            fieldnames = ["state_id", "most_important"] + \
                         [f"imp_{fname}" for fname in feature_indices]
            with open(self.output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in per_state_results:
                    writer.writerow(row)
            print(f"\nPer-state results saved to {self.output_file}")

    def get_label_names(self) -> list:
        return [f"imp_{fname}" for fname in self.features_to_test] + ["imp_none"]

    def has_dynamic_labels(self) -> bool:
        return True
