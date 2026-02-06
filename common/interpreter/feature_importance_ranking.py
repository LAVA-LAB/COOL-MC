import csv
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from common.interpreter.interpreter import Interpreter
from common.agents.subset_decision_tree_agent import SubsetDecisionTreeAgent


class FeatureImportanceRankingInterpreter(Interpreter):

    def __init__(self, config):
        super().__init__(config)
        parts = config.split(";")
        self.test_size = float(parts[1]) if len(parts) > 1 else 0.5

    def _run_direction(self, direction_label, ordered_indices, feature_names,
                       X_train, X_test, y_train, y_test,
                       env, num_actions, constant_definitions, property_query,
                       max_depth=None):
        n_features = len(ordered_indices)
        results = []

        print(f"\n{'='*80}")
        print(f"Direction: {direction_label}  (max_depth={max_depth})")
        print(f"{'='*80}")

        for k in range(1, n_features + 1):
            selected_indices = ordered_indices[:k]
            selected_names = [feature_names[i] for i in selected_indices]

            # Train decision tree on feature subset with depth limit
            X_train_subset = X_train[:, selected_indices]
            X_test_subset = X_test[:, selected_indices]
            clf = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            clf.fit(X_train_subset, y_train)

            y_train_pred = clf.predict(X_train_subset)
            y_test_pred = clf.predict(X_test_subset)
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            features_str = ", ".join(selected_names)

            print(f"\n{'-'*80}")
            print(f"Iteration {k}/{n_features}  |  Features: [{features_str}]")
            print(f"Train Accuracy: {train_acc*100:.2f}%  |  Test Accuracy: {test_acc*100:.2f}%")

            if test_acc == 1.0:
                print(">>> 100% test accuracy reached with this feature combination! <<<")

            wrapper_agent = SubsetDecisionTreeAgent(
                clf, selected_indices, num_actions)

            try:
                mdp_result, mc_info = env.storm_bridge.model_checker.induced_markov_chain(
                    wrapper_agent, None, env,
                    constant_definitions, property_query)
                print(f"Property result: {mdp_result}")
            except Exception as e:
                print(f"Model checking failed: {e}")
                mdp_result = None

            results.append({
                'direction': direction_label,
                'property_query': property_query,
                'property_result': mdp_result,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'used_features': features_str,
                'n_features': k,
            })

        return results

    def interpret(self, env, rl_agent, model_checking_info):
        # ----------------------------------------------------------------
        # 1. Prepare data
        # ----------------------------------------------------------------
        X = np.array(model_checking_info['collected_states'])
        y = np.array(model_checking_info['collected_action_idizes'])
        property_query = model_checking_info['property']
        constant_definitions = env.storm_bridge.constant_definitions
        action_labels = env.action_mapper.actions
        num_actions = len(action_labels)

        # Feature names
        state_mapper = env.storm_bridge.state_mapper
        if state_mapper.has_compressed_state_representation():
            feature_names = state_mapper.get_compressed_feature_names()
            if feature_names is None:
                feature_names = state_mapper.get_feature_names()
        else:
            feature_names = state_mapper.get_feature_names()

        n_samples, n_features = X.shape
        print(f"\n{'='*80}")
        print("Feature Importance Ranking Interpreter")
        print(f"{'='*80}")
        print(f"Samples: {n_samples}  Features: {n_features}  Actions: {num_actions}")
        print(f"Property: {property_query}")
        print(f"Constant definitions: {constant_definitions}")

        # ----------------------------------------------------------------
        # 2. Train/test split
        # ----------------------------------------------------------------
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=True, random_state=42)
        print(f"Train: {len(X_train)}  Test: {len(X_test)}  (test_size={self.test_size})")

        # ----------------------------------------------------------------
        # 3. Determine max_depth from the original agent's tree
        # ----------------------------------------------------------------
        # Cap depth so the tree cannot memorize the training set
        n_train = len(X_train)
        depth_cap = int(np.ceil(np.log2(max(n_train, 2))))

        max_depth = None
        if hasattr(rl_agent, 'classifier') and hasattr(rl_agent.classifier, 'get_depth'):
            original_depth = rl_agent.classifier.get_depth()
            max_depth = min(original_depth, depth_cap)
            print(f"Using max_depth={max_depth} (original agent depth={original_depth}, cap={depth_cap} from {n_train} train samples)")
        else:
            max_depth = depth_cap
            print(f"Using max_depth={max_depth} (cap=log2({n_train}))")

        # ----------------------------------------------------------------
        # 4. Train full decision tree to obtain Gini importances
        # ----------------------------------------------------------------
        full_clf = tree.DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        full_clf.fit(X_train, y_train)
        importances = full_clf.feature_importances_

        # Rank features by importance (descending)
        best_to_worst = np.argsort(importances)[::-1]
        worst_to_best = np.argsort(importances)

        ranked_names = [feature_names[i] for i in best_to_worst]
        ranked_importances = importances[best_to_worst]

        print(f"\nFeature importance ranking (Gini):")
        for rank, (idx, name, imp) in enumerate(
                zip(best_to_worst, ranked_names, ranked_importances), 1):
            print(f"  {rank:>3}. {name:<30s}  importance={imp:.6f}")

        # ----------------------------------------------------------------
        # 5. Worst-to-best direction
        # ----------------------------------------------------------------
        results_wtb = self._run_direction(
            "Worst to Best", worst_to_best, feature_names,
            X_train, X_test, y_train, y_test,
            env, num_actions, constant_definitions, property_query,
            max_depth=max_depth)

        # ----------------------------------------------------------------
        # 6. Best-to-worst direction
        # ----------------------------------------------------------------
        results_btw = self._run_direction(
            "Best to Worst", best_to_worst, feature_names,
            X_train, X_test, y_train, y_test,
            env, num_actions, constant_definitions, property_query,
            max_depth=max_depth)

        # ----------------------------------------------------------------
        # 6. Print final summary tables
        # ----------------------------------------------------------------
        all_results = results_btw + results_wtb

        for direction, results in [("Worst to Best", results_wtb),
                                   ("Best to Worst", results_btw)]:
            results_sorted = sorted(
                results,
                key=lambda r: (r['property_result'] is not None,
                               r['property_result'] if r['property_result'] is not None else 0),
                reverse=True)

            print(f"\n{'='*80}")
            print(f"SUMMARY — {direction}  (sorted by property result descending)")
            print(f"{'='*80}")
            header = f"{'#':>3}  {'Train Acc':>10}  {'Test Acc':>10}  {'Property Result':>16}  Features"
            print(header)
            print("-" * len(header) + "-" * 40)
            for i, r in enumerate(results_sorted, 1):
                pr = f"{r['property_result']:.6f}" if r['property_result'] is not None else "N/A"
                print(f"{i:>3}  {r['train_accuracy']*100:>9.2f}%  {r['test_accuracy']*100:>9.2f}%  {pr:>16s}  {r['used_features']}")

        # ----------------------------------------------------------------
        # 7. Save to CSV
        # ----------------------------------------------------------------
        all_sorted = sorted(
            all_results,
            key=lambda r: (r['direction'],
                           r['property_result'] is not None,
                           r['property_result'] if r['property_result'] is not None else 0),
            reverse=True)

        csv_path = 'feature_importance_ranking_results.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'direction', 'property_query', 'property_result',
                'train_accuracy', 'test_accuracy', 'n_features', 'used_features'])
            writer.writeheader()
            for r in all_sorted:
                writer.writerow(r)
        print(f"\nResults saved to {csv_path}")
