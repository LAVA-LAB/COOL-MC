import json
import re
import numpy as np
from common.state_labelers.state_labeler import StateLabeler


class FeatureRangeLabeler(StateLabeler):
    """Labels states based on whether feature values fall in specified ranges.

    Enables PCTL queries about clinically meaningful patient subgroups,
    e.g., P=? [ "high_sofa" U "survival" ] to check whether patients with
    high organ failure scores can survive under the current policy.

    Configuration format:
        "feature_range;feature=[min,max]:label;feature=[min,max]:label;..."

    Supports absolute values or percentile-based ranges (p0..p100):
        "feature_range;SOFA=[10,24]:high_sofa"           # absolute range
        "feature_range;SOFA=[p75,p100]:high_sofa"        # top 25% of SOFA values
        "feature_range;GCS=[p0,p25]:low_gcs"             # bottom 25% of GCS values

    For each range specification, creates two labels:
        - label_name (e.g., "high_sofa") for states where feature is in range
        - not_label_name (e.g., "not_high_sofa") for states where feature is out of range

    Example:
        "feature_range;SOFA=[p75,p100]:high_sofa;Arterial_lactate=[p75,p100]:high_lactate"
    """

    def __init__(self, config_str: str):
        self.range_specs = []  # [(feature_name, lo_str, hi_str, label_name), ...]
        super().__init__(config_str)

    def parse_config(self, config_str: str) -> None:
        parts = config_str.split(";")
        for part in parts[1:]:
            part = part.strip()
            if not part:
                continue
            # Parse: feature_name=[min,max]:label_name
            match = re.match(r'(\w+)=\[([^,]+),([^\]]+)\]:(\w+)', part)
            if not match:
                raise ValueError(
                    f"Invalid range spec: '{part}'. "
                    f"Expected: feature=[min,max]:label")
            feature_name = match.group(1)
            lo_str = match.group(2).strip()
            hi_str = match.group(3).strip()
            label_name = match.group(4)
            self.range_specs.append((feature_name, lo_str, hi_str, label_name))

        print(f"FeatureRangeLabeler initialized with {len(self.range_specs)} range(s):")
        for feat, lo, hi, label in self.range_specs:
            print(f"  {feat}=[{lo},{hi}] -> \"{label}\" / \"not_{label}\"")

    def mark_state_before_preprocessing(self, state: np.ndarray, agent, state_json: str) -> None:
        """Collect decompressed feature vectors during incremental building."""
        if state_json not in self.collected_data:
            self.collected_data[state_json] = state.copy()

    def _resolve_value(self, val_str, all_values):
        """Resolve a value string to float. Supports percentiles like 'p75'."""
        val_str = val_str.strip()
        if val_str.startswith('p'):
            percentile = float(val_str[1:])
            return np.percentile(all_values, percentile)
        return float(val_str)

    def label_states(self, model, env, agent) -> None:
        """Add feature-range labels to the Storm model."""
        # Get feature names
        state_mapper = env.storm_bridge.state_mapper
        if state_mapper.has_compressed_state_representation():
            feature_names = state_mapper.get_compressed_feature_names()
        else:
            feature_names = state_mapper.get_feature_names()

        if feature_names is None:
            n_features = len(next(iter(self.collected_data.values())))
            feature_names = [f"f{i}" for i in range(n_features)]

        # Build state_id -> features mapping
        state_features = {}
        for state_id in range(model.nr_states):
            state_json = str(model.state_valuations.get_json(state_id))
            if state_json in self.collected_data:
                state_features[state_id] = self.collected_data[state_json]
            else:
                # Fallback: decompress on the fly
                try:
                    state_dict = json.loads(state_json)
                    example_json = json.loads(env.storm_bridge.state_json_example)
                    filtered = {k: v for k, v in state_dict.items()
                                if k in example_json}
                    state_features[state_id] = env.storm_bridge.parse_state(
                        json.dumps(filtered))
                except Exception:
                    state_features[state_id] = None

        # Collect states with valid features
        valid_ids = [sid for sid in state_features if state_features[sid] is not None]
        if not valid_ids:
            print("FeatureRangeLabeler: No states with features found.")
            return

        feature_matrix = np.array([state_features[sid] for sid in valid_ids])

        # Print feature statistics
        print(f"\nFeature statistics across {len(valid_ids)} reachable states:")
        print(f"{'Feature':<25} {'Min':>10} {'P25':>10} {'Median':>10} "
              f"{'P75':>10} {'Max':>10}")
        print("-" * 80)
        for i, name in enumerate(feature_names):
            vals = feature_matrix[:, i]
            print(f"{name:<25} {vals.min():>10.2f} {np.percentile(vals, 25):>10.2f} "
                  f"{np.percentile(vals, 50):>10.2f} {np.percentile(vals, 75):>10.2f} "
                  f"{vals.max():>10.2f}")

        # Process each range specification
        for feat_name, lo_str, hi_str, label_name in self.range_specs:
            if feat_name not in feature_names:
                print(f"WARNING: Feature '{feat_name}' not found. "
                      f"Available: {feature_names}")
                continue

            feat_idx = feature_names.index(feat_name)
            all_vals = feature_matrix[:, feat_idx]

            lo = self._resolve_value(lo_str, all_vals)
            hi = self._resolve_value(hi_str, all_vals)

            pos_label = label_name
            neg_label = f"not_{label_name}"
            if pos_label not in model.labeling.get_labels():
                model.labeling.add_label(pos_label)
            if neg_label not in model.labeling.get_labels():
                model.labeling.add_label(neg_label)

            pos_count = 0
            neg_count = 0

            for state_id in range(model.nr_states):
                feat_vec = state_features.get(state_id)
                if feat_vec is not None:
                    val = feat_vec[feat_idx]
                    if lo <= val <= hi:
                        model.labeling.add_label_to_state(pos_label, state_id)
                        pos_count += 1
                    else:
                        model.labeling.add_label_to_state(neg_label, state_id)
                        neg_count += 1
                else:
                    # Terminal/absorbing states get negative label
                    model.labeling.add_label_to_state(neg_label, state_id)
                    neg_count += 1

            resolved = f"[{lo:.4f}, {hi:.4f}]"
            if lo_str.startswith('p') or hi_str.startswith('p'):
                resolved += f" (from [{lo_str}, {hi_str}])"

            print(f"\n  {feat_name} in {resolved} -> \"{pos_label}\":")
            print(f"    {pos_label}: {pos_count} states "
                  f"({pos_count / model.nr_states * 100:.1f}%)")
            print(f"    {neg_label}: {neg_count} states "
                  f"({neg_count / model.nr_states * 100:.1f}%)")

    def has_dynamic_labels(self) -> bool:
        return True
