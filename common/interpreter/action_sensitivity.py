import csv
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from common.interpreter.interpreter import Interpreter
from common.behavioral_cloning_dataset.all_optimal_dataset import AllOptimalDataset


class ActionSensitivityInterpreter(Interpreter):
    """
    Action Sensitivity Interpreter for multi-dimensional action spaces.

    Uses the all_optimal_dataset from model checking to determine how
    constrained each action dimension is per state. For instance, in ICU
    sepsis with actions f{i}_v{j} (fluid level x vasopressor level), this
    reveals in which states the fluid dosage choice is critical versus the
    vasopressor dosage.

    For each state, importance of dimension d is measured as:
        importance_d = 1 - (n_optimal_levels_d / n_total_levels_d)

    If only 1 out of 5 vasopressor levels is optimal, importance = 0.8
    (getting it right matters a lot). If all 5 fluid levels are optimal,
    importance = 0.0 (fluids don't matter in that state).

    Config format: "action_sensitivity;dim1_name,dim2_name[;filter_expr][;output_file.csv]"

    The dimension names map positionally to the "_"-separated parts of
    action labels. E.g. for actions "f0_v0", "f1_v2", ...
        "action_sensitivity;fluids,vasopressors"
    means the first part (f0..f4) is "fluids" and the second (v0..v4)
    is "vasopressors".

    Examples:
        "action_sensitivity;fluids,vasopressors"
        "action_sensitivity;fluids,vasopressors;SOFA=[0,5]:GCS=[3,8]"
        "action_sensitivity;fluids,vasopressors;;results.csv"
        "action_sensitivity;fluids,vasopressors;SOFA=[0,5];results.csv"
    """

    def __init__(self, config):
        super().__init__(config)
        parts = config.split(";")
        self.dim_names = [n.strip() for n in parts[1].split(",")]
        self.filter_expr = parts[2].strip() if len(parts) > 2 and parts[2].strip() else ""
        self.output_file = parts[3].strip() if len(parts) > 3 and parts[3].strip() else ""

    # -- helpers --

    def _get_feature_names(self, env):
        """Get feature names, preferring compressed names."""
        state_mapper = env.storm_bridge.state_mapper
        if state_mapper.has_compressed_state_representation():
            names = state_mapper.get_compressed_feature_names()
            if names is not None:
                return names
        return state_mapper.get_feature_names()

    def _parse_filters(self, filter_expr, feature_names):
        """Parse filter like 'SOFA=[0,10]:GCS=[3,8]' into {idx: (lo, hi)}."""
        if not filter_expr:
            return {}
        filters = {}
        for entry in filter_expr.split(":"):
            entry = entry.strip()
            if not entry:
                continue
            name, range_str = entry.split("=", 1)
            name = name.strip()
            if name not in feature_names:
                raise ValueError(
                    f"Filter feature '{name}' not found. "
                    f"Available features: {feature_names}")
            idx = feature_names.index(name)
            range_str = range_str.strip().strip("[]")
            lo, hi = range_str.split(",")
            filters[idx] = (float(lo.strip()), float(hi.strip()))
        return filters

    def _apply_filters(self, X, y, filters):
        """Keep only states matching all filter constraints."""
        if not filters:
            return X, y
        mask = np.ones(len(X), dtype=bool)
        for idx, (lo, hi) in filters.items():
            mask &= (X[:, idx] >= lo) & (X[:, idx] <= hi)
        return X[mask], y[mask]

    # -- action dimension parsing --

    def _parse_action_dimensions(self, action_labels):
        """Parse action labels like 'f0_v0' into dimension structure.

        Returns:
            n_dims: number of dimensions
            dim_levels: dict[dim_idx] -> sorted list of unique levels
            action_to_levels: dict[action_idx] -> tuple of levels
        """
        n_dims = None
        dim_level_sets = {}
        action_to_levels = {}

        for action_idx, label in enumerate(action_labels):
            parts = label.split("_")
            if n_dims is None:
                n_dims = len(parts)
                for d in range(n_dims):
                    dim_level_sets[d] = set()
            elif len(parts) != n_dims:
                raise ValueError(
                    f"Action '{label}' has {len(parts)} dimensions, "
                    f"expected {n_dims} (based on '{action_labels[0]}')")

            levels = []
            for d, part in enumerate(parts):
                match = re.search(r'(\d+)$', part)
                level = int(match.group(1)) if match else part
                dim_level_sets[d].add(level)
                levels.append(level)
            action_to_levels[action_idx] = tuple(levels)

        dim_levels = {d: sorted(lvls) for d, lvls in dim_level_sets.items()}
        return n_dims, dim_levels, action_to_levels

    # -- build optimal dataset --

    def _build_optimal_dataset(self, env, property_query):
        """Build the all_optimal_dataset to get all optimal actions per state."""
        prism_path = env.storm_bridge.path
        constant_defs = env.storm_bridge.constant_definitions

        config = f"all_optimal_dataset;{prism_path};{property_query};{constant_defs}"
        dataset = AllOptimalDataset(config)
        dataset.create(env)
        data = dataset.get_data()
        return data['X_train'], data['y_train']

    # -- importance computation --

    def _compute_importance(self, state_to_optimal_actions, action_to_levels,
                            n_dims, dim_levels):
        """Compute per-state, per-dimension importance from optimal actions.

        For dimension d in state s:
          importance = 1 - (n_optimal_levels_d / n_total_levels_d)

        Returns array of shape [n_states, n_dims] and list of state keys.
        """
        state_keys = list(state_to_optimal_actions.keys())
        n_states = len(state_keys)
        importance = np.zeros((n_states, n_dims))

        for s_idx, state_key in enumerate(state_keys):
            optimal_actions = state_to_optimal_actions[state_key]

            for d in range(n_dims):
                n_total = len(dim_levels[d])
                # Collect unique optimal levels for this dimension
                optimal_levels_d = set()
                for action_idx in optimal_actions:
                    if action_idx in action_to_levels:
                        optimal_levels_d.add(action_to_levels[action_idx][d])

                n_optimal = len(optimal_levels_d)
                importance[s_idx, d] = 1.0 - (n_optimal / n_total)

        return importance, state_keys

    # -- main --

    def interpret(self, env, rl_agent, model_checking_info):
        property_query = model_checking_info['property']

        feature_names = self._get_feature_names(env)
        action_labels = env.action_mapper.actions

        # Parse action dimensions
        n_dims, dim_levels, action_to_levels = self._parse_action_dimensions(action_labels)

        if len(self.dim_names) != n_dims:
            raise ValueError(
                f"Provided {len(self.dim_names)} dimension names {self.dim_names} "
                f"but actions have {n_dims} dimensions "
                f"(e.g. '{action_labels[0]}' splits into {n_dims} parts)")

        # Header
        print(f"\n{'='*80}")
        print("Action Sensitivity Interpreter (Optimal Action Analysis)")
        print(f"{'='*80}")
        print(f"Property:          {property_query}")
        print(f"Action dimensions: {n_dims}")
        for d in range(n_dims):
            print(f"  {self.dim_names[d]}: "
                  f"{len(dim_levels[d])} levels {dim_levels[d]}")
        print(f"Total actions:     {len(action_labels)}")

        # Build all_optimal_dataset
        print(f"\nBuilding all-optimal dataset...")
        X_opt, y_opt = self._build_optimal_dataset(env, property_query)
        print(f"Optimal state-action pairs: {len(X_opt)}")

        # Parse & apply state filters
        filters = self._parse_filters(self.filter_expr, feature_names)
        X_filtered, y_filtered = self._apply_filters(X_opt, y_opt, filters)

        # Group by state: state_tuple -> set of optimal action indices
        state_to_optimal_actions = defaultdict(set)
        for i in range(len(X_filtered)):
            state_key = tuple(X_filtered[i])
            state_to_optimal_actions[state_key].add(y_filtered[i])

        n_unique_states = len(state_to_optimal_actions)

        if filters:
            for idx, (lo, hi) in filters.items():
                print(f"  Filter: {feature_names[idx]} in [{lo}, {hi}]")
        else:
            print(f"State filter:      none (all states)")
        print(f"Unique states:     {n_unique_states}")

        if n_unique_states == 0:
            print("No states match the filter. Aborting.")
            return

        # Compute per-state, per-dimension importance
        importance, state_keys = self._compute_importance(
            state_to_optimal_actions, action_to_levels, n_dims, dim_levels)

        # Mean optimal actions per state
        n_optimal_per_state = [len(state_to_optimal_actions[k]) for k in state_keys]
        print(f"Mean optimal actions per state: {np.mean(n_optimal_per_state):.2f}")

        # ----------------------------------------------------------------
        # Overall statistics
        # ----------------------------------------------------------------
        mean_imp = importance.mean(axis=0)
        std_imp = importance.std(axis=0)

        print(f"\n{'='*80}")
        print("Overall Importance (1 = only one level is optimal, 0 = all levels are optimal)")
        print(f"{'='*80}")
        header = f"{'Dimension':<25s}  {'Mean Importance':>16s}  {'Std':>12s}"
        print(header)
        print("-" * len(header))
        for d in range(n_dims):
            print(f"{self.dim_names[d]:<25s}  {mean_imp[d]:>16.4f}  {std_imp[d]:>12.4f}")

        # ----------------------------------------------------------------
        # Dominant dimension per state
        # ----------------------------------------------------------------
        dominant = np.argmax(importance, axis=1)

        print(f"\nMost constrained dimension per state:")
        for d in range(n_dims):
            count = int(np.sum(dominant == d))
            pct = count / n_unique_states * 100
            print(f"  {self.dim_names[d]}: {count} states ({pct:.1f}%)")

        # Count tied states (equal importance across all dims)
        tied = np.all(importance == importance[:, 0:1], axis=1)
        n_tied = int(np.sum(tied))
        if n_tied > 0:
            print(f"  (tied/equal: {n_tied} states ({n_tied/n_unique_states*100:.1f}%))")

        # ----------------------------------------------------------------
        # Ratio analysis (2D case)
        # ----------------------------------------------------------------
        if n_dims == 2:
            total_imp = importance.sum(axis=1)
            nonzero = total_imp > 1e-10
            if nonzero.any():
                ratio = np.zeros(n_unique_states)
                ratio[nonzero] = importance[nonzero, 0] / total_imp[nonzero]
                print(f"\n{self.dim_names[0]} dominance ratio "
                      f"(0 = only {self.dim_names[1]} constrained, "
                      f"1 = only {self.dim_names[0]} constrained):")
                print(f"  Mean:   {ratio[nonzero].mean():.4f}")
                print(f"  Median: {np.median(ratio[nonzero]):.4f}")
                print(f"  Std:    {ratio[nonzero].std():.4f}")

        # ----------------------------------------------------------------
        # Top states per dimension
        # ----------------------------------------------------------------
        n_top = min(5, n_unique_states)
        for d in range(n_dims):
            top_indices = np.argsort(importance[:, d])[::-1][:n_top]
            print(f"\nTop {n_top} states where {self.dim_names[d]} "
                  f"matters most:")
            print(f"  {'State':>6s}  {'Importance':>12s}  "
                  f"{'#Optimal Acts':>14s}  {'Optimal Levels':<30s}")
            print(f"  {'-'*6}  {'-'*12}  {'-'*14}  {'-'*30}")
            for si in top_indices:
                state_key = state_keys[si]
                opt_actions = state_to_optimal_actions[state_key]
                # Show which levels are optimal for this dim
                opt_levels_d = sorted(set(
                    action_to_levels[a][d] for a in opt_actions
                    if a in action_to_levels))
                print(f"  {si:>6d}  {importance[si, d]:>12.4f}  "
                      f"{len(opt_actions):>14d}  "
                      f"{self.dim_names[d]}={opt_levels_d}")

        # ----------------------------------------------------------------
        # Save CSV
        # ----------------------------------------------------------------
        csv_path = (self.output_file if self.output_file
                    else 'action_sensitivity_results.csv')
        self._save_csv(importance, dominant, state_to_optimal_actions,
                       state_keys, action_to_levels, action_labels,
                       feature_names, n_dims, property_query, csv_path)

        # ----------------------------------------------------------------
        # Save plot
        # ----------------------------------------------------------------
        if csv_path.endswith('.csv'):
            plot_path = csv_path[:-4] + '.png'
        else:
            plot_path = 'action_sensitivity_results.png'

        if n_dims == 2:
            self._save_scatter_plot(
                importance, self.dim_names, plot_path, self.filter_expr)
        else:
            self._save_bar_plot(
                mean_imp, std_imp, self.dim_names, plot_path, self.filter_expr)

    def _save_csv(self, importance, dominant, state_to_optimal_actions,
                  state_keys, action_to_levels, action_labels,
                  feature_names, n_dims, property_query, csv_path):
        """Save per-state results to CSV."""
        imp_cols = [f'{name}_importance' for name in self.dim_names]
        fieldnames = (['state_index'] + imp_cols +
                      ['dominant_dimension', 'n_optimal_actions',
                       'property_query', 'filter'])

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for s in range(len(state_keys)):
                row = {'state_index': s}
                for d in range(n_dims):
                    row[imp_cols[d]] = float(importance[s, d])
                row['dominant_dimension'] = self.dim_names[dominant[s]]
                row['n_optimal_actions'] = len(
                    state_to_optimal_actions[state_keys[s]])
                row['property_query'] = property_query
                row['filter'] = self.filter_expr if self.filter_expr else 'none'
                writer.writerow(row)
        print(f"\nResults saved to {csv_path}")

    @staticmethod
    def _save_scatter_plot(importance, dim_names, plot_path, filter_expr):
        """Save scatter plot for 2D action spaces."""
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(importance[:, 0], importance[:, 1], alpha=0.5, s=20)

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'r--', alpha=0.5,
                label='Equal importance')

        ax.set_xlabel(f'{dim_names[0]} importance')
        ax.set_ylabel(f'{dim_names[1]} importance')
        title = f'Action Sensitivity: {dim_names[0]} vs {dim_names[1]}'
        if filter_expr:
            title += f'\nFilter: {filter_expr}'
        ax.set_title(title)
        ax.legend()
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved to {plot_path}")

    @staticmethod
    def _save_bar_plot(mean_imp, std_imp, dim_names, plot_path, filter_expr):
        """Save bar chart for N-dimensional action spaces."""
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(dim_names))
        ax.bar(x, mean_imp, yerr=std_imp, alpha=0.8, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels(dim_names)
        ax.set_ylabel('Mean Importance')
        ax.set_ylim(0, 1.05)
        title = 'Action Dimension Importance (Optimal Action Constraint)'
        if filter_expr:
            title += f'\nFilter: {filter_expr}'
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved to {plot_path}")
