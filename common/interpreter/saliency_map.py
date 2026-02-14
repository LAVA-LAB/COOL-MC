import csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common.interpreter.interpreter import Interpreter


class SaliencyMapInterpreter(Interpreter):
    """
    Saliency Map Feature Importance Ranking for neural network agents (DQN and PPO).

    Computes input gradient saliency maps to rank features by their importance
    to the agent's policy decisions. Optionally filters to a state subspace
    so that importance is measured only within a specific region.

    The interpreter:
    1. Collects states visited during model checking
    2. Optionally filters states to a user-defined subspace
    3. Computes |d output / d input| for each state (gradient-based saliency)
    4. Averages saliency across states to rank features
    5. Saves ranking table to CSV and bar chart to PNG

    Config format: "saliency_map[;feature1=[min,max]:feature2=[min,max]][;output_file.csv]"

    Filter syntax uses ":" to separate feature constraints:
        SOFA=[0,10]:SpO2=[90,100]

    Examples:
        "saliency_map"                                       - all states
        "saliency_map;SOFA=[0,10]:SpO2=[90,100]"            - filtered subspace
        "saliency_map;SOFA=[0,10];my_results.csv"           - filtered + custom output
        "saliency_map;;my_results.csv"                       - all states + custom output
    """

    def __init__(self, config):
        super().__init__(config)
        parts = config.split(";")
        self.filter_expr = parts[1].strip() if len(parts) > 1 and parts[1].strip() else ""
        self.output_file = parts[2].strip() if len(parts) > 2 and parts[2].strip() else ""

    def _get_feature_names(self, env):
        """Get feature names, preferring compressed names."""
        state_mapper = env.storm_bridge.state_mapper
        if state_mapper.has_compressed_state_representation():
            names = state_mapper.get_compressed_feature_names()
            if names is not None:
                return names
        return state_mapper.get_feature_names()

    def _parse_filters(self, filter_expr, feature_names):
        """Parse filter expression like 'SOFA=[0,10]:Arterial_lactate=[0,5]'.

        Returns dict mapping feature index -> (min_val, max_val).
        """
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
            min_val, max_val = range_str.split(",")
            filters[idx] = (float(min_val.strip()), float(max_val.strip()))
        return filters

    def _apply_filters(self, X, y, filters):
        """Keep only states matching all filter constraints."""
        if not filters:
            return X, y
        mask = np.ones(len(X), dtype=bool)
        for idx, (lo, hi) in filters.items():
            mask &= (X[:, idx] >= lo) & (X[:, idx] <= hi)
        return X[mask], y[mask]

    def _get_network(self, agent):
        """Return (network, agent_type_str)."""
        if hasattr(agent, 'q_eval'):
            return agent.q_eval, "DQN"
        elif hasattr(agent, 'policy'):
            return agent.policy, "PPO"
        raise ValueError(
            f"Unsupported agent type: {type(agent).__name__}. "
            f"Expected DQNAgent (q_eval) or PPOAgent (policy).")

    def _compute_logits(self, network, states_t, agent_type):
        """Forward pass returning logits (before softmax) for both agent types."""
        if agent_type == "DQN":
            # DQN forward returns raw Q-values (no softmax)
            return network(states_t)
        else:
            # PPO: compute logits without softmax to avoid saturation
            x = states_t
            for layer in network.layers:
                x = F.relu(layer(x))
            return network.output(x)

    def _compute_saliency(self, network, states, actions, agent_type):
        """Compute absolute input gradients for each state/action pair.

        Returns numpy array of shape [n_states, n_features].
        """
        network.eval()

        device = next(network.parameters()).device
        states_t = torch.tensor(
            states, dtype=torch.float32, device=device, requires_grad=True)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)

        # Forward pass to logits (no softmax)
        logits = self._compute_logits(network, states_t, agent_type)

        # Gather the logit for the selected action per state
        selected = logits.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Backward to get gradients w.r.t. input
        selected.sum().backward()

        saliency = states_t.grad.abs().detach().cpu().numpy()
        return saliency

    def interpret(self, env, rl_agent, model_checking_info):
        X = np.array(model_checking_info['collected_states'])
        y = np.array(model_checking_info['collected_action_idizes'])
        property_query = model_checking_info['property']

        feature_names = self._get_feature_names(env)
        n_total, n_features = X.shape

        # Parse & apply state subspace filters
        filters = self._parse_filters(self.filter_expr, feature_names)
        X_filtered, y_filtered = self._apply_filters(X, y, filters)
        n_filtered = len(X_filtered)

        print(f"\n{'='*80}")
        print("Saliency Map Feature Importance Ranking")
        print(f"{'='*80}")
        print(f"Property:        {property_query}")
        print(f"Total states:    {n_total}")

        if filters:
            for idx, (lo, hi) in filters.items():
                print(f"  Filter: {feature_names[idx]} in [{lo}, {hi}]")
        else:
            print(f"State filter:    none (all states)")

        print(f"Filtered states: {n_filtered}")

        if n_filtered == 0:
            print("No states match the filter. Aborting.")
            return

        # Action distribution in the filtered subspace
        unique_actions, action_counts = np.unique(y_filtered, return_counts=True)
        action_labels = env.action_mapper.actions
        print(f"\nAction distribution in subspace:")
        for a, c in zip(unique_actions, action_counts):
            label = action_labels[a] if a < len(action_labels) else str(a)
            print(f"  {label}: {c} ({c/n_filtered*100:.1f}%)")

        # Get network
        network, agent_type = self._get_network(rl_agent)
        print(f"\nAgent type:      {agent_type}")

        # Compute saliency
        saliency = self._compute_saliency(network, X_filtered, y_filtered, agent_type)

        # Mean and std absolute saliency per feature
        mean_saliency = saliency.mean(axis=0)
        std_saliency = saliency.std(axis=0)

        # Rank descending by mean saliency
        ranking = np.argsort(mean_saliency)[::-1]

        print(f"\n{'='*80}")
        print("Feature Importance Ranking (by mean |gradient|)")
        print(f"{'='*80}")
        header = f"{'Rank':>4}  {'Feature':<30s}  {'Mean |grad|':>12s}  {'Std |grad|':>12s}"
        print(header)
        print("-" * len(header))

        results = []
        for rank, idx in enumerate(ranking, 1):
            name = feature_names[idx]
            mean_val = mean_saliency[idx]
            std_val = std_saliency[idx]
            print(f"{rank:>4}  {name:<30s}  {mean_val:>12.6f}  {std_val:>12.6f}")
            results.append({
                'rank': rank,
                'feature': name,
                'feature_index': int(idx),
                'mean_abs_gradient': float(mean_val),
                'std_abs_gradient': float(std_val),
                'n_states': n_filtered,
                'property_query': property_query,
                'filter': self.filter_expr if self.filter_expr else 'none',
            })

        # Save CSV
        csv_path = self.output_file if self.output_file else 'saliency_map_results.csv'
        fieldnames = ['rank', 'feature', 'feature_index', 'mean_abs_gradient',
                      'std_abs_gradient', 'n_states', 'property_query', 'filter']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nResults saved to {csv_path}")

        # Save bar plot
        if csv_path.endswith('.csv'):
            plot_path = csv_path[:-4] + '.png'
        else:
            plot_path = 'saliency_map_results.png'
        self._save_plot(feature_names, mean_saliency, std_saliency,
                        ranking, plot_path, self.filter_expr)

    @staticmethod
    def _save_plot(feature_names, mean_saliency, std_saliency,
                   ranking, plot_path, filter_expr):
        """Save a horizontal bar chart of feature saliency."""
        names = [feature_names[i] for i in ranking]
        means = [mean_saliency[i] for i in ranking]
        stds = [std_saliency[i] for i in ranking]

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
        y_pos = np.arange(len(names))
        ax.barh(y_pos, means, xerr=stds, align='center', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |gradient|')
        title = 'Saliency Map Feature Importance'
        if filter_expr:
            title += f'\nFilter: {filter_expr}'
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved to {plot_path}")
