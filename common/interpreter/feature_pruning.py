import csv
import numpy as np
import torch
from common.interpreter.interpreter import Interpreter


class FeaturePruningInterpreter(Interpreter):
    """
    Feature Pruning Interpreter for neural network agents (DQN and PPO).

    Prunes specified input features from the first layer of the neural network
    and re-runs model checking to measure the impact on the safety property.

    The interpreter:
    1. Runs a baseline model check on the unpruned network
    2. Zeros out all outgoing weights from the specified input neurons in the first layer
    3. Runs model checking on the pruned network
    4. Reports the change in the property result compared to the baseline

    Config format: "feature_pruning;feature1,feature2,...[;output_file.csv]"

    Use "all" to prune every feature individually:
        "feature_pruning;all"  - prunes each feature one at a time
    """

    def __init__(self, config):
        super().__init__(config)
        parts = config.split(";")
        raw = parts[1].strip()
        self.prune_all = (raw.lower() == "all")
        self.feature_names_to_prune = [] if self.prune_all else [f.strip() for f in raw.split(",")]
        self.output_file = parts[2].strip() if len(parts) > 2 and parts[2].strip() else ""

    def _get_first_layer(self, agent):
        """Detect agent type and return the first linear layer of the network."""
        # DQN agent: network is agent.q_eval, layers is a Sequential
        if hasattr(agent, 'q_eval'):
            return agent.q_eval.layers[0], "DQN"
        # PPO agent: network is agent.policy, layers is a Sequential
        elif hasattr(agent, 'policy'):
            return agent.policy.layers[0], "PPO"
        else:
            raise ValueError(
                f"Unsupported agent type: {type(agent).__name__}. "
                f"Expected DQNAgent (q_eval) or PPOAgent (policy).")

    def _feature_names_to_indices(self, env, feature_names):
        """Map feature names to their indices in the state vector.

        For compressed state models, looks up indices in the compressed
        feature names list. Otherwise uses the PRISM state mapper.
        """
        state_mapper = env.storm_bridge.state_mapper
        indices = []

        if state_mapper.has_compressed_state_representation():
            compressed_names = state_mapper.get_compressed_feature_names()
            if compressed_names is not None:
                for name in feature_names:
                    if name in compressed_names:
                        indices.append(compressed_names.index(name))
                    else:
                        raise ValueError(
                            f"Feature '{name}' not found in compressed feature names. "
                            f"Available features: {compressed_names}")
                return indices

        for name in feature_names:
            if name in state_mapper.mapper:
                indices.append(state_mapper.mapper[name])
            else:
                available = list(state_mapper.mapper.keys())
                raise ValueError(
                    f"Feature '{name}' not found in state mapper. "
                    f"Available features: {available}")
        return indices

    def _prune_and_get_original(self, layer, feature_indices):
        """Zero out weights for specified input features. Returns cloned originals."""
        original_weight = layer.weight.data.clone()
        for idx in feature_indices:
            layer.weight.data[:, idx] = 0
        return original_weight

    def _restore_weights(self, layer, original_weight):
        """Restore original weights after pruning."""
        layer.weight.data = original_weight

    def _run_model_checking(self, agent, env, constant_definitions, property_query):
        """Run model checking and return (result, info)."""
        try:
            mdp_result, mc_info = env.storm_bridge.model_checker.induced_markov_chain(
                agent, None, env, constant_definitions, property_query)
            return mdp_result, mc_info
        except Exception as e:
            print(f"  Model checking failed: {e}")
            return None, None

    def _get_all_feature_names(self, env):
        """Get all feature names, preferring compressed names."""
        state_mapper = env.storm_bridge.state_mapper
        if state_mapper.has_compressed_state_representation():
            names = state_mapper.get_compressed_feature_names()
            if names is not None:
                return names
        return state_mapper.get_feature_names()

    def interpret(self, env, rl_agent, model_checking_info):
        property_query = model_checking_info['property']
        constant_definitions = env.storm_bridge.constant_definitions

        # Detect agent type and get the first layer
        first_layer, agent_type = self._get_first_layer(rl_agent)

        # Resolve "all" to every available feature
        all_feature_names = self._get_all_feature_names(env)
        if self.prune_all:
            self.feature_names_to_prune = list(all_feature_names)

        # Map feature names to indices
        feature_indices = self._feature_names_to_indices(env, self.feature_names_to_prune)

        print(f"\n{'='*80}")
        print("Feature Pruning Interpreter")
        print(f"{'='*80}")
        print(f"Agent type:        {agent_type}")
        print(f"Property:          {property_query}")
        print(f"Mode:              {'all features individually' if self.prune_all else 'specified features together'}")
        print(f"Features to prune: {self.feature_names_to_prune}")
        print(f"Total features:    {first_layer.in_features}")

        # ----------------------------------------------------------------
        # 1. Baseline (unpruned network)
        # ----------------------------------------------------------------
        print(f"\n{'-'*80}")
        print("Running baseline model checking (no pruning)...")
        baseline_result, _ = self._run_model_checking(
            rl_agent, env, constant_definitions, property_query)
        print(f"Baseline property result: {baseline_result}")

        results = []
        results.append({
            'pruned_features': 'none (baseline)',
            'property_query': property_query,
            'property_result': baseline_result,
            'absolute_change': 0.0,
            'relative_change_pct': 0.0,
        })

        if self.prune_all:
            # --------------------------------------------------------------
            # 2a. "all" mode: prune each feature individually
            # --------------------------------------------------------------
            print(f"\n{'='*80}")
            print("Individual Feature Pruning (all features)")
            print(f"{'='*80}")

            for fname, fidx in zip(self.feature_names_to_prune, feature_indices):
                print(f"\n{'-'*80}")
                print(f"Pruning feature: {fname} (index {fidx})")

                original_weight = self._prune_and_get_original(first_layer, [fidx])
                pruned_result, mc_info = self._run_model_checking(
                    rl_agent, env, constant_definitions, property_query)
                self._restore_weights(first_layer, original_weight)

                abs_change, rel_change = self._compute_change(baseline_result, pruned_result)

                print(f"  Property result: {pruned_result}")
                if abs_change is not None:
                    print(f"  Change from baseline: {abs_change:+.6f} ({rel_change:+.2f}%)")

                results.append({
                    'pruned_features': fname,
                    'property_query': property_query,
                    'property_result': pruned_result,
                    'absolute_change': abs_change,
                    'relative_change_pct': rel_change,
                })
        else:
            # --------------------------------------------------------------
            # 2b. Normal mode: prune specified features together
            # --------------------------------------------------------------
            pruned_names = ", ".join(self.feature_names_to_prune)
            print(f"\n{'='*80}")
            print(f"Pruning features: [{pruned_names}]")
            print(f"{'='*80}")

            original_weight = self._prune_and_get_original(first_layer, feature_indices)
            pruned_result, mc_info = self._run_model_checking(
                rl_agent, env, constant_definitions, property_query)
            self._restore_weights(first_layer, original_weight)

            abs_change, rel_change = self._compute_change(baseline_result, pruned_result)

            print(f"  Property result: {pruned_result}")
            if abs_change is not None:
                print(f"  Change from baseline: {abs_change:+.6f} ({rel_change:+.2f}%)")
            if mc_info is not None:
                print(f"  Model Size: {mc_info['model_size']}")
                print(f"  Transitions: {mc_info['model_transitions']}")

            results.append({
                'pruned_features': pruned_names,
                'property_query': property_query,
                'property_result': pruned_result,
                'absolute_change': abs_change,
                'relative_change_pct': rel_change,
            })

        # ----------------------------------------------------------------
        # 3. Summary
        # ----------------------------------------------------------------
        self._print_summary(results)

        # ----------------------------------------------------------------
        # 4. Save to CSV
        # ----------------------------------------------------------------
        csv_path = self.output_file if self.output_file else 'feature_pruning_results.csv'
        self._save_csv(results, csv_path)

    @staticmethod
    def _compute_change(baseline, pruned):
        """Compute absolute and relative change from baseline."""
        if pruned is None or baseline is None:
            return None, None
        abs_change = pruned - baseline
        if baseline != 0:
            rel_change = abs_change / abs(baseline) * 100
        else:
            rel_change = float('inf') if abs_change != 0 else 0.0
        return abs_change, rel_change

    @staticmethod
    def _print_summary(results):
        """Print a summary table of all pruning results."""
        print(f"\n{'='*80}")
        print("SUMMARY - Feature Pruning Impact")
        print(f"{'='*80}")
        header = (f"{'Pruned Feature':<35s}  {'Property Result':>16s}  "
                  f"{'Abs. Change':>12s}  {'Rel. Change':>12s}")
        print(header)
        print("-" * len(header))

        for r in results:
            pr = f"{r['property_result']:.6f}" if r['property_result'] is not None else "N/A"
            ac = f"{r['absolute_change']:+.6f}" if r['absolute_change'] is not None else "N/A"
            rc = f"{r['relative_change_pct']:+.2f}%" if r['relative_change_pct'] is not None else "N/A"
            print(f"{r['pruned_features']:<35s}  {pr:>16s}  {ac:>12s}  {rc:>12s}")

        # Impact ranking (non-baseline entries)
        pruned = [r for r in results if r['pruned_features'] != 'none (baseline)']
        if len(pruned) > 1:
            ranked = sorted(
                pruned,
                key=lambda r: abs(r['absolute_change']) if r['absolute_change'] is not None else 0,
                reverse=True)
            print(f"\nFeature Impact Ranking (by absolute change):")
            for rank, r in enumerate(ranked, 1):
                ac = f"{r['absolute_change']:+.6f}" if r['absolute_change'] is not None else "N/A"
                print(f"  {rank}. {r['pruned_features']}: {ac}")

    @staticmethod
    def _save_csv(results, csv_path):
        """Save results to a CSV file."""
        fieldnames = ['pruned_features', 'property_query', 'property_result',
                      'absolute_change', 'relative_change_pct']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nResults saved to {csv_path}")
