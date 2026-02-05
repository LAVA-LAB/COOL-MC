"""
Behavioral Cloning Agent using Decision Tree Ensemble with Majority Voting.

This agent trains multiple decision trees with DIVERSE hyperparameters
and uses majority voting for prediction. Diversity is achieved through:
  1. Different random seeds for each tree
  2. Varied max_depth across trees
  3. Varied min_samples_leaf across trees
  4. Varied splitter strategies (best vs random)

This diversity ensures that even when trained on the same data, different
trees will learn different decision boundaries, creating a truly permissive
ensemble policy.
"""
import os
import shutil
import joblib
import mlflow
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from scipy import stats
from common.agents.agent import Agent


class BCDecisionTreeEnsembleAgent(Agent):
    """
    Behavioral Cloning agent using an ensemble of Decision Trees with majority voting.

    Creates diverse trees through varied hyperparameters:
    - Different random seeds
    - Varied max_depth (base ± variation)
    - Varied min_samples_leaf
    - Mix of 'best' and 'random' splitters

    This diversity is crucial for creating a permissive policy where different
    trees may predict different actions for the same state.
    """

    def __init__(self, state_dimension: int, number_of_actions: int,
                 n_estimators: int = 10, max_depth: int = None,
                 min_samples_leaf: int = 1):
        """Initialize BC Decision Tree Ensemble Agent.

        Args:
            state_dimension (int): State dimension
            number_of_actions (int): Number of actions
            n_estimators (int, optional): Number of trees in ensemble. Defaults to 10.
            max_depth (int, optional): Base maximum depth of each tree. None for unlimited.
            min_samples_leaf (int, optional): Base minimum samples per leaf. Defaults to 1.
        """
        self.state_dimension = state_dimension
        self.number_of_actions = number_of_actions
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        # Create ensemble of decision trees with DIVERSE hyperparameters
        self.trees = []
        self.tree_configs = []  # Store configs for reporting

        for i in range(n_estimators):
            # Vary the random seed
            seed = 42 + i * 17  # More spread out seeds

            # Vary max_depth: if base is None (unlimited), use varied depths
            # Otherwise vary around the base value
            if max_depth is None or max_depth == 0:
                # Use varied depths: some shallow, some deep, some unlimited
                depth_options = [None, 5, 8, 10, 12, 15, 20, None, 25, None]
                tree_max_depth = depth_options[i % len(depth_options)]
            else:
                # Vary around the base: base-2 to base+3
                variation = (i % 6) - 2  # -2, -1, 0, 1, 2, 3
                tree_max_depth = max(2, max_depth + variation)

            # Vary min_samples_leaf: 1, 1, 2, 1, 2, 3, 1, 2, 1, 2
            leaf_options = [1, 1, 2, 1, 2, 3, 1, 2, 1, 2]
            tree_min_samples = max(1, min_samples_leaf + leaf_options[i % len(leaf_options)] - 1)

            # Vary splitter: mix of 'best' and 'random'
            splitter = 'random' if i % 3 == 0 else 'best'

            # Vary max_features: None (all), 'sqrt', 'log2'
            max_features_options = [None, 'sqrt', 'log2', None, None]
            max_features = max_features_options[i % len(max_features_options)]

            config = {
                'max_depth': tree_max_depth,
                'min_samples_leaf': tree_min_samples,
                'splitter': splitter,
                'max_features': max_features,
                'random_state': seed
            }
            self.tree_configs.append(config)

            self.trees.append(
                tree.DecisionTreeClassifier(**config)
            )

        self.is_trained = False

    def behavioral_cloning(self, env, data: dict, epochs: int = 1, accuracy_threshold: float = 100.0):
        """
        Train the decision tree ensemble on behavioral cloning data.

        Note: epochs and accuracy_threshold parameters are accepted for API
        consistency but not used (decision trees train in a single pass).

        Args:
            env: Environment (used for feature names)
            data: Dict with 'X_train', 'y_train', 'X_test', 'y_test'
            epochs: Ignored for decision trees
            accuracy_threshold: Ignored for decision trees (single-pass training)

        Returns:
            Tuple of (epoch, train_acc, test_acc, train_loss, test_loss)
        """
        X_train = data.get('X_train')
        y_train = data.get('y_train')
        X_test = data.get('X_test')
        y_test = data.get('y_test')

        if X_train is None or y_train is None:
            return None, None, None, None, None

        # Print class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\nClass distribution in training data:")
        for cls, count in zip(unique, counts):
            print(f"  Action {cls}: {count} samples")

        # Train each tree on the full dataset with DIVERSE hyperparameters
        print(f"\nTraining Diverse Decision Tree Ensemble (n_estimators={self.n_estimators})...")
        print("Diversity through: varied max_depth, min_samples_leaf, splitter, max_features, seeds")

        tree_stats = []
        for i, dt in enumerate(self.trees):
            dt.fit(X_train, y_train)
            tree_stats.append({
                'depth': dt.get_depth(),
                'n_leaves': dt.get_n_leaves(),
                'config': self.tree_configs[i]
            })

        self.is_trained = True

        # Compute training accuracy using majority voting
        y_train_pred = self._predict_majority_vote(X_train)
        train_accuracy = 100 * accuracy_score(y_train, y_train_pred)
        n_correct = (y_train_pred == y_train).sum()
        n_total = len(y_train)
        n_wrong = n_total - n_correct

        # Compute test accuracy if available
        test_accuracy = None
        if X_test is not None and y_test is not None:
            y_test_pred = self._predict_majority_vote(X_test)
            test_accuracy = 100 * accuracy_score(y_test, y_test_pred)

        # Compute diversity metrics
        depths = [s['depth'] for s in tree_stats]
        leaves = [s['n_leaves'] for s in tree_stats]

        # Check prediction diversity on training data
        all_predictions = np.array([dt.predict(X_train) for dt in self.trees])  # (n_trees, n_samples)
        # Count how many samples have disagreement among trees
        n_disagreements = 0
        for j in range(X_train.shape[0]):
            unique_preds = len(set(all_predictions[:, j]))
            if unique_preds > 1:
                n_disagreements += 1
        diversity_rate = 100 * n_disagreements / X_train.shape[0]

        # Print ensemble statistics
        print(f"\n{'='*70}")
        print(f"Diverse Decision Tree Ensemble Training Complete!")
        print(f"{'='*70}")
        print(f"Ensemble Statistics:")
        print(f"  Number of trees:    {self.n_estimators}")
        print(f"  Depth range:        {min(depths)} - {max(depths)} (avg: {np.mean(depths):.1f})")
        print(f"  Leaves range:       {min(leaves)} - {max(leaves)} (avg: {np.mean(leaves):.1f})")
        print(f"  Train Accuracy:     {train_accuracy:.4f}% ({n_correct}/{n_total} correct, {n_wrong} wrong)")
        print(f"  Tree Disagreement:  {diversity_rate:.1f}% of training samples have diverse predictions")
        if test_accuracy is not None:
            print(f"  Test Accuracy:      {test_accuracy:.2f}%")
        print(f"{'='*70}\n")

        return 0, train_accuracy, test_accuracy, 0.0, 0.0

    def _predict_majority_vote(self, X: np.ndarray) -> np.ndarray:
        """Get predictions using majority voting across all trees.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Array of predicted class labels
        """
        # Get predictions from all trees
        predictions = np.array([dt.predict(X) for dt in self.trees])  # (n_trees, n_samples)

        # Majority vote (mode along tree axis)
        majority_votes, _ = stats.mode(predictions, axis=0, keepdims=False)
        return majority_votes.astype(int)

    def select_action(self, state: np.ndarray, deploy: bool = False) -> int:
        """Select action using majority voting from the ensemble.

        Args:
            state (np.ndarray): Current state
            deploy (bool, optional): Ignored for this agent. Defaults to False.

        Returns:
            int: Selected action
        """
        if not self.is_trained:
            return np.random.randint(0, self.number_of_actions)

        state_2d = state.reshape(1, -1) if state.ndim == 1 else state
        action = self._predict_majority_vote(state_2d)[0]
        return int(action)

    def get_ensemble_actions(self, state: np.ndarray) -> list[int]:
        """Get individual actions from all decision trees in the ensemble.

        Args:
            state (np.ndarray): Current state

        Returns:
            list[int]: List of actions, one from each tree in the ensemble

        Raises:
            RuntimeError: If the ensemble has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble has not been trained yet.")

        state_2d = state.reshape(1, -1) if state.ndim == 1 else state
        return [int(dt.predict(state_2d)[0]) for dt in self.trees]

    def get_raw_outputs(self, state: np.ndarray) -> np.ndarray:
        """Get vote proportions for each action (soft voting).

        Args:
            state (np.ndarray): Current state

        Returns:
            np.ndarray: Proportion of trees voting for each action
        """
        if not self.is_trained:
            return np.ones(self.number_of_actions) / self.number_of_actions

        state_2d = state.reshape(1, -1) if state.ndim == 1 else state

        # Count votes for each action
        votes = np.zeros(self.number_of_actions)
        for dt in self.trees:
            pred = dt.predict(state_2d)[0]
            if pred < self.number_of_actions:
                votes[int(pred)] += 1

        # Normalize to proportions
        return votes / self.n_estimators

    def save(self, artifact_path='model'):
        """Save the ensemble to MLFlow artifacts."""
        try:
            os.makedirs('tmp_model', exist_ok=True)
        except Exception:
            pass

        # Save all trees
        for i, dt in enumerate(self.trees):
            joblib.dump(dt, f'tmp_model/bc_dt_ensemble_tree_{i}.joblib')

        # Save metadata including tree configs for diversity
        metadata = {
            'state_dimension': self.state_dimension,
            'number_of_actions': self.number_of_actions,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'is_trained': self.is_trained,
            'tree_configs': self.tree_configs
        }
        joblib.dump(metadata, 'tmp_model/bc_dt_ensemble_meta.joblib')

        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)
        shutil.rmtree('tmp_model')

    def load(self, model_root_folder_path: str):
        """Load the ensemble from disk.

        Args:
            model_root_folder_path (str): Path to model folder
        """
        try:
            # Load metadata first
            meta_path = os.path.join(model_root_folder_path, 'bc_dt_ensemble_meta.joblib')
            if os.path.exists(meta_path):
                metadata = joblib.load(meta_path)
                self.n_estimators = metadata.get('n_estimators', self.n_estimators)
                self.max_depth = metadata.get('max_depth', self.max_depth)
                self.min_samples_leaf = metadata.get('min_samples_leaf', self.min_samples_leaf)
                # Load tree configs if available
                if 'tree_configs' in metadata:
                    self.tree_configs = metadata['tree_configs']

            # Check if any tree files exist before clearing
            first_tree_path = os.path.join(model_root_folder_path, 'bc_dt_ensemble_tree_0.joblib')
            if not os.path.exists(first_tree_path):
                # No saved trees found, keep the initialized trees for training
                print(f"No saved ensemble trees found at {model_root_folder_path}, using fresh trees")
                return

            # Load all trees
            loaded_trees = []
            for i in range(self.n_estimators):
                tree_path = os.path.join(model_root_folder_path, f'bc_dt_ensemble_tree_{i}.joblib')
                if os.path.exists(tree_path):
                    loaded_trees.append(joblib.load(tree_path))
                else:
                    print(f"Warning: Tree {i} not found at {tree_path}")

            if len(loaded_trees) == self.n_estimators:
                self.trees = loaded_trees
                self.is_trained = True
                print(f"Successfully loaded Diverse Decision Tree Ensemble ({self.n_estimators} trees) from {model_root_folder_path}")
            else:
                print(f"Warning: Only loaded {len(loaded_trees)}/{self.n_estimators} trees, keeping fresh trees")

        except Exception as msg:
            print(f"Error loading Decision Tree Ensemble agent: {msg}")

    def get_hyperparameters(self):
        """Get agent hyperparameters."""
        return {
            'state_dimension': self.state_dimension,
            'number_of_actions': self.number_of_actions,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf
        }

    def get_tree_agreement(self, state: np.ndarray) -> float:
        """Get the proportion of trees that agree on the prediction.

        Useful for understanding ensemble confidence.

        Args:
            state: Input state

        Returns:
            Float between 0 and 1 indicating agreement level
        """
        if not self.is_trained:
            return 0.0

        state_2d = state.reshape(1, -1) if state.ndim == 1 else state
        predictions = [dt.predict(state_2d)[0] for dt in self.trees]

        # Find most common prediction and its count
        unique, counts = np.unique(predictions, return_counts=True)
        max_count = counts.max()

        return max_count / self.n_estimators
