import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from common.interpreter.interpreter import Interpreter
from common.behavioral_cloning_dataset.all_optimal_dataset import AllOptimalDataset


class ActionDistributionInterpreter(Interpreter):
    """
    Action Distribution Interpreter comparing optimal action distributions
    across two PCTL properties.

    For each property, an all_optimal behavioral cloning dataset is built.
    The number of optimal action occurrences is counted per action and
    normalized to percentages (relative to the total for that property).
    A grouped bar chart is produced showing the distribution side by side.

    Config format:
        "action_distribution;prop1;prop2[;output_file.csv]"

    Examples:
        "action_distribution;P=? [ F \"goal\" ];P=? [ F \"fail\" ]"
        "action_distribution;P=? [ F \"goal\" ];P=? [ F \"fail\" ];results.csv"
    """

    def __init__(self, config):
        super().__init__(config)
        parts = config.split(";")
        if len(parts) < 3:
            raise ValueError(
                "action_distribution requires at least two properties. "
                'Format: "action_distribution;prop1;prop2[;output_file.csv]"')
        self.prop1 = parts[1].strip()
        self.prop2 = parts[2].strip()
        self.output_file = parts[3].strip() if len(parts) > 3 and parts[3].strip() else ""

    def _build_optimal_dataset(self, env, property_query):
        """Build an all_optimal_dataset for the given property."""
        prism_path = env.storm_bridge.path
        constant_defs = env.storm_bridge.constant_definitions
        config = f"all_optimal_dataset;{prism_path};{property_query};{constant_defs}"
        dataset = AllOptimalDataset(config)
        dataset.create(env)
        data = dataset.get_data()
        return data['X_train'], data['y_train']

    def interpret(self, env, rl_agent, model_checking_info):
        action_labels = env.action_mapper.actions
        n_actions = len(action_labels)

        print(f"\n{'='*80}")
        print("Action Distribution Interpreter")
        print(f"{'='*80}")
        print(f"Property 1: {self.prop1}")
        print(f"Property 2: {self.prop2}")
        print(f"Actions:    {n_actions} ({', '.join(action_labels)})")

        # Build datasets for both properties
        print(f"\nBuilding all-optimal dataset for property 1...")
        _, y1 = self._build_optimal_dataset(env, self.prop1)
        print(f"  Optimal state-action pairs: {len(y1)}")

        print(f"Building all-optimal dataset for property 2...")
        _, y2 = self._build_optimal_dataset(env, self.prop2)
        print(f"  Optimal state-action pairs: {len(y2)}")

        # Count action occurrences
        counts1 = Counter(y1)
        counts2 = Counter(y2)

        total1 = sum(counts1.values())
        total2 = sum(counts2.values())

        # Compute normalized percentages for each action
        pct1 = np.array([counts1.get(i, 0) / total1 * 100 for i in range(n_actions)])
        pct2 = np.array([counts2.get(i, 0) / total2 * 100 for i in range(n_actions)])

        # Compute relative share per action: count1 / (count1 + count2)
        raw1 = np.array([counts1.get(i, 0) for i in range(n_actions)], dtype=float)
        raw2 = np.array([counts2.get(i, 0) for i in range(n_actions)], dtype=float)
        combined = raw1 + raw2
        share1 = np.where(combined > 0, raw1 / combined * 100, 0.0)
        share2 = np.where(combined > 0, raw2 / combined * 100, 0.0)

        # Print results table
        print(f"\n{'='*80}")
        print("Action Distribution (normalized per property)")
        print(f"{'='*80}")
        header = f"{'Action':<20s}  {'Prop 1 (%)':>10s}  {'Prop 2 (%)':>10s}"
        print(header)
        print("-" * len(header))
        for i in range(n_actions):
            print(f"{action_labels[i]:<20s}  {pct1[i]:>10.2f}  {pct2[i]:>10.2f}")

        print(f"\n{'='*80}")
        print("Relative Share per Action (property1 vs property2)")
        print(f"{'='*80}")
        header2 = (f"{'Action':<20s}  {'Prop 1 cnt':>10s}  {'Prop 2 cnt':>10s}  "
                   f"{'Prop 1 (%)':>10s}  {'Prop 2 (%)':>10s}")
        print(header2)
        print("-" * len(header2))
        for i in range(n_actions):
            print(f"{action_labels[i]:<20s}  {int(raw1[i]):>10d}  {int(raw2[i]):>10d}  "
                  f"{share1[i]:>10.2f}  {share2[i]:>10.2f}")

        # Save CSV
        csv_path = self.output_file if self.output_file else 'action_distribution_results.csv'
        self._save_csv(action_labels, pct1, pct2, counts1, counts2,
                       share1, share2, n_actions, csv_path)

        # Save plots
        if csv_path.endswith('.csv'):
            plot_path = csv_path[:-4] + '.png'
            share_plot_path = csv_path[:-4] + '_relative.png'
        else:
            plot_path = 'action_distribution_results.png'
            share_plot_path = 'action_distribution_results_relative.png'
        self._save_plot(action_labels, pct1, pct2, n_actions, plot_path)
        self._save_share_plot(action_labels, share1, share2, n_actions, share_plot_path)

    def _save_csv(self, action_labels, pct1, pct2, counts1, counts2,
                  share1, share2, n_actions, csv_path):
        """Save action distribution results to CSV."""
        fieldnames = ['action', 'property1_count', 'property1_pct',
                      'property2_count', 'property2_pct',
                      'relative_share_property1', 'relative_share_property2',
                      'property1', 'property2']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(n_actions):
                writer.writerow({
                    'action': action_labels[i],
                    'property1_count': counts1.get(i, 0),
                    'property1_pct': f"{pct1[i]:.2f}",
                    'property2_count': counts2.get(i, 0),
                    'property2_pct': f"{pct2[i]:.2f}",
                    'relative_share_property1': f"{share1[i]:.2f}",
                    'relative_share_property2': f"{share2[i]:.2f}",
                    'property1': self.prop1,
                    'property2': self.prop2,
                })
        print(f"\nResults saved to {csv_path}")

    def _save_plot(self, action_labels, pct1, pct2, n_actions, plot_path):
        """Save grouped bar chart comparing action distributions."""
        x = np.arange(n_actions)
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, n_actions * 0.8), 6))

        bars1 = ax.bar(x - bar_width / 2, pct1, bar_width, label=self.prop1)
        bars2 = ax.bar(x + bar_width / 2, pct2, bar_width, label=self.prop2)

        ax.set_xlabel('Action')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Action Distribution per Property')
        ax.set_xticks(x)
        ax.set_xticklabels(action_labels, rotation=45, ha='right')
        ax.legend()

        # Add value labels on bars
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=7)
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords='offset points',
                            ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Plot saved to {plot_path}")

    def _save_share_plot(self, action_labels, share1, share2, n_actions, plot_path):
        """Save stacked 100% bar chart showing relative share per action."""
        x = np.arange(n_actions)
        bar_width = 0.6

        fig, ax = plt.subplots(figsize=(max(8, n_actions * 0.8), 6))

        bars1 = ax.bar(x, share1, bar_width, label=self.prop1)
        bars2 = ax.bar(x, share2, bar_width, bottom=share1, label=self.prop2)

        ax.set_xlabel('Action')
        ax.set_ylabel('Relative Share (%)')
        ax.set_title('Relative Action Distribution (Property 1 vs Property 2)')
        ax.set_xticks(x)
        ax.set_xticklabels(action_labels, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color='black', linestyle='--', linewidth=1, alpha=0.7, label='50%')
        ax.legend()

        # Add percentage labels inside bars
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            if share1[i] > 5:
                ax.text(bar1.get_x() + bar1.get_width() / 2, share1[i] / 2,
                        f'{share1[i]:.1f}%', ha='center', va='center', fontsize=7)
            if share2[i] > 5:
                ax.text(bar2.get_x() + bar2.get_width() / 2,
                        share1[i] + share2[i] / 2,
                        f'{share2[i]:.1f}%', ha='center', va='center', fontsize=7)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Relative share plot saved to {plot_path}")
