"""
Behavioral Cloning Agent using a Mixed Neural Network + Decision Tree Ensemble
with Majority Voting.

This agent trains an ensemble where half the members are neural networks and
half are decision trees. When the total number of estimators N is odd, the
extra member is a decision tree.

During retraining, neural network weights are transferred from the previous
model and retrained, while decision trees are trained fresh on the current
dataset.
"""
import os
import shutil
import joblib
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from sklearn import tree
from sklearn.metrics import accuracy_score
from scipy import stats
from common.agents.agent import Agent
from common.agents.bc_nn_ensemble_agent import EnsembleNetwork
from common.utilities.helper import DEVICE


class BCMixedEnsembleAgent(Agent):
    """
    Behavioral Cloning agent using a mixed ensemble of Neural Networks and
    Decision Trees with majority voting.

    Given n_estimators=N:
      - n_nn = N // 2 neural networks
      - n_dt = N - n_nn decision trees (gets the extra one when N is odd)

    Neural networks provide diversity through different random seeds.
    Decision trees provide diversity through varied hyperparameters
    (max_depth, min_samples_leaf, splitter, max_features).
    """

    def __init__(self, state_dimension: int, number_of_neurons: List[int],
                 number_of_actions: int, n_estimators: int = 5,
                 learning_rate: float = 0.001, batch_size: int = 64,
                 max_depth: int = None, min_samples_leaf: int = 1):
        """Initialize BC Mixed Ensemble Agent.

        Args:
            state_dimension: State dimension
            number_of_neurons: Neurons per hidden layer (for NN part)
            number_of_actions: Number of actions
            n_estimators: Total number of ensemble members. Defaults to 5.
            learning_rate: Learning rate for NNs. Defaults to 0.001.
            batch_size: Batch size for NN training. Defaults to 64.
            max_depth: Base max depth for DTs. None for unlimited.
            min_samples_leaf: Base min samples per leaf for DTs. Defaults to 1.
        """
        self.state_dimension = state_dimension
        self.number_of_neurons = number_of_neurons
        self.number_of_actions = number_of_actions
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        # Split: half NNs, remaining DTs (odd N gives extra DT)
        self.n_nn = n_estimators // 2
        self.n_dt = n_estimators - self.n_nn

        # Create neural networks with different random seeds
        self.networks = [
            EnsembleNetwork(
                state_dimension, number_of_neurons, number_of_actions,
                learning_rate, seed=42 + i
            )
            for i in range(self.n_nn)
        ]

        # Create decision trees with diverse hyperparameters
        self.trees = []
        self.tree_configs = []
        for i in range(self.n_dt):
            seed = 42 + i * 17

            if max_depth is None or max_depth == 0:
                depth_options = [None, 5, 8, 10, 12, 15, 20, None, 25, None]
                tree_max_depth = depth_options[i % len(depth_options)]
            else:
                variation = (i % 6) - 2
                tree_max_depth = max(2, max_depth + variation)

            leaf_options = [1, 1, 2, 1, 2, 3, 1, 2, 1, 2]
            tree_min_samples = max(1, min_samples_leaf + leaf_options[i % len(leaf_options)] - 1)

            splitter = 'random' if i % 3 == 0 else 'best'

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
            self.trees.append(tree.DecisionTreeClassifier(**config))

        self.is_trained = False

    def behavioral_cloning(self, env, data: dict, epochs: int = 100, accuracy_threshold: float = 100.0):
        """
        Train the mixed ensemble on behavioral cloning data.

        Decision trees are trained first (single pass). Then neural networks
        are trained epoch-by-epoch with early stopping.

        Args:
            env: Environment
            data: Dict with 'X_train', 'y_train', 'X_test', 'y_test'
            epochs: Number of training epochs (for NNs)
            accuracy_threshold: Stop NN training early when all networks >= this (percent)

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

        print(f"\nTraining Mixed Ensemble (n_estimators={self.n_estimators}: "
              f"{self.n_nn} NNs + {self.n_dt} DTs)")

        # --- Train Decision Trees (single pass) ---
        if self.n_dt > 0:
            print(f"\n--- Training {self.n_dt} Decision Trees ---")
            tree_stats = []
            for i, dt in enumerate(self.trees):
                dt.fit(X_train, y_train)
                tree_stats.append({
                    'depth': dt.get_depth(),
                    'n_leaves': dt.get_n_leaves(),
                    'config': self.tree_configs[i]
                })

            depths = [s['depth'] for s in tree_stats]
            leaves = [s['n_leaves'] for s in tree_stats]
            print(f"  DT depth range: {min(depths)} - {max(depths)} (avg: {np.mean(depths):.1f})")
            print(f"  DT leaves range: {min(leaves)} - {max(leaves)} (avg: {np.mean(leaves):.1f})")

        # --- Train Neural Networks (epoch-by-epoch) ---
        final_epoch = 0
        if self.n_nn > 0:
            print(f"\n--- Training {self.n_nn} Neural Networks ---")
            print(f"  Architecture: {self.number_of_neurons}, lr={self.learning_rate}")
            print(f"  Early stopping when all NNs >= {accuracy_threshold}%")

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
            num_samples = X_train_tensor.shape[0]

            # Compute class weights
            unique_classes, class_counts = torch.unique(y_train_tensor, return_counts=True)
            class_weights = torch.ones(self.number_of_actions, dtype=torch.float32)
            for cls, count in zip(unique_classes, class_counts):
                class_weights[cls] = num_samples / (self.number_of_actions * count.item())
            class_weights = class_weights.to(DEVICE)
            weighted_loss = nn.CrossEntropyLoss(weight=class_weights)

            best_accuracies = [0.0] * self.n_nn
            temp_checkpoints = [f'tmp_mixed_nn_best_{i}.chkpt' for i in range(self.n_nn)]

            final_epoch = epochs - 1
            for epoch in range(epochs):
                net_accs = []
                for net_idx, network in enumerate(self.networks):
                    network.train()

                    indices = torch.randperm(num_samples)
                    X_shuffled = X_train_tensor[indices]
                    y_shuffled = y_train_tensor[indices]

                    correct = 0
                    total = 0

                    for i in range(0, num_samples, self.batch_size):
                        batch_X = X_shuffled[i:i + self.batch_size]
                        batch_y = y_shuffled[i:i + self.batch_size]

                        network.optimizer.zero_grad()
                        outputs = network.forward(batch_X)
                        loss = weighted_loss(outputs, batch_y)
                        loss.backward()
                        network.optimizer.step()

                        _, predicted = torch.max(outputs.data, 1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()

                    net_accuracy = 100 * correct / total
                    net_accs.append(net_accuracy)

                    if net_accuracy > best_accuracies[net_idx]:
                        best_accuracies[net_idx] = net_accuracy
                        network.save_checkpoint(temp_checkpoints[net_idx])

                if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                    avg_acc = np.mean(net_accs)
                    print(f"  Epoch {epoch + 1}/{epochs}: Avg NN Acc={avg_acc:.2f}% "
                          f"(individual: {', '.join(f'{a:.1f}%' for a in net_accs)})")

                if all(acc >= accuracy_threshold for acc in net_accs):
                    print(f"\n  Early stopping at epoch {epoch + 1}/{epochs}: "
                          f"all NNs >= {accuracy_threshold}%")
                    final_epoch = epoch
                    break

            # Load best checkpoint for each network
            for i in range(self.n_nn):
                if os.path.exists(temp_checkpoints[i]):
                    self.networks[i].load_checkpoint(temp_checkpoints[i])
                    os.remove(temp_checkpoints[i])
                print(f"  NN {i + 1} best accuracy: {best_accuracies[i]:.2f}%")

        self.is_trained = True

        # Compute ensemble accuracy using majority voting
        y_train_pred = self._predict_majority_vote(X_train)
        train_accuracy = 100 * np.mean(y_train_pred == y_train)

        test_accuracy = None
        if X_test is not None and y_test is not None:
            y_test_pred = self._predict_majority_vote(X_test)
            test_accuracy = 100 * np.mean(y_test_pred == y_test)

        print(f"\n{'='*70}")
        print(f"Mixed Ensemble Training Complete!")
        print(f"{'='*70}")
        print(f"Ensemble Statistics:")
        print(f"  Total members:       {self.n_estimators} ({self.n_nn} NNs + {self.n_dt} DTs)")
        if self.n_nn > 0:
            print(f"  NN Architecture:     {self.number_of_neurons}")
            print(f"  NN Epochs completed: {final_epoch + 1}/{epochs}")
        print(f"  Ensemble Train Accuracy: {train_accuracy:.2f}%")
        if test_accuracy is not None:
            print(f"  Ensemble Test Accuracy:  {test_accuracy:.2f}%")
        print(f"{'='*70}\n")

        return final_epoch, train_accuracy, test_accuracy, 0.0, 0.0

    def _predict_majority_vote(self, X: np.ndarray) -> np.ndarray:
        """Get predictions using majority voting across all NNs and DTs.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Array of predicted class labels
        """
        all_predictions = []

        # NN predictions
        if self.n_nn > 0:
            X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            for network in self.networks:
                network.eval()
                with torch.no_grad():
                    outputs = network.forward(X_tensor)
                    _, predicted = torch.max(outputs, 1)
                    all_predictions.append(predicted.cpu().numpy())

        # DT predictions
        for dt in self.trees:
            all_predictions.append(dt.predict(X))

        predictions = np.array(all_predictions)  # (n_estimators, n_samples)
        majority_votes, _ = stats.mode(predictions, axis=0, keepdims=False)
        return majority_votes.astype(int)

    def select_action(self, state: np.ndarray, deploy: bool = False) -> int:
        """Select action using majority voting from the mixed ensemble.

        Args:
            state: Current state
            deploy: Ignored for this agent. Defaults to False.

        Returns:
            Selected action
        """
        if not self.is_trained:
            return np.random.randint(0, self.number_of_actions)

        state_2d = state.reshape(1, -1) if state.ndim == 1 else state
        action = self._predict_majority_vote(state_2d)[0]
        return int(action)

    def get_ensemble_actions(self, state: np.ndarray) -> list[int]:
        """Get individual actions from all ensemble members.

        Args:
            state: Current state

        Returns:
            List of actions, one from each NN then each DT

        Raises:
            RuntimeError: If the ensemble has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble has not been trained yet.")

        state_2d = state.reshape(1, -1) if state.ndim == 1 else state
        actions = []

        # NN actions
        state_tensor = torch.tensor(state_2d, dtype=torch.float32).to(DEVICE)
        for network in self.networks:
            network.eval()
            with torch.no_grad():
                outputs = network.forward(state_tensor)
                _, predicted = torch.max(outputs, 1)
                actions.append(int(predicted.cpu().item()))

        # DT actions
        for dt in self.trees:
            actions.append(int(dt.predict(state_2d)[0]))

        return actions

    def get_raw_outputs(self, state: np.ndarray) -> np.ndarray:
        """Get vote proportions for each action (soft voting).

        Args:
            state: Current state

        Returns:
            Proportion of ensemble members voting for each action
        """
        if not self.is_trained:
            return np.ones(self.number_of_actions) / self.number_of_actions

        actions = self.get_ensemble_actions(state)
        votes = np.zeros(self.number_of_actions)
        for action in actions:
            if action < self.number_of_actions:
                votes[action] += 1

        return votes / self.n_estimators

    def save(self, artifact_path='model'):
        """Save the mixed ensemble to MLFlow artifacts."""
        try:
            os.makedirs('tmp_model', exist_ok=True)
        except Exception:
            pass

        # Save NN checkpoints
        for i, network in enumerate(self.networks):
            network.save_checkpoint(f'tmp_model/bc_mixed_ensemble_nn_{i}.chkpt')

        # Save DT trees
        for i, dt in enumerate(self.trees):
            joblib.dump(dt, f'tmp_model/bc_mixed_ensemble_dt_{i}.joblib')

        # Save metadata
        metadata = {
            'state_dimension': self.state_dimension,
            'number_of_neurons': self.number_of_neurons,
            'number_of_actions': self.number_of_actions,
            'n_estimators': self.n_estimators,
            'n_nn': self.n_nn,
            'n_dt': self.n_dt,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'is_trained': self.is_trained,
            'tree_configs': self.tree_configs
        }
        torch.save(metadata, 'tmp_model/bc_mixed_ensemble_meta.pt')

        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)
        shutil.rmtree('tmp_model')

    def load(self, model_root_folder_path: str):
        """Load the mixed ensemble from disk.

        Args:
            model_root_folder_path: Path to model folder
        """
        try:
            # Load metadata
            meta_path = os.path.join(model_root_folder_path, 'bc_mixed_ensemble_meta.pt')
            if os.path.exists(meta_path):
                metadata = torch.load(meta_path, map_location=DEVICE)
                self.n_estimators = metadata.get('n_estimators', self.n_estimators)
                self.n_nn = metadata.get('n_nn', self.n_nn)
                self.n_dt = metadata.get('n_dt', self.n_dt)
                self.learning_rate = metadata.get('learning_rate', self.learning_rate)
                self.batch_size = metadata.get('batch_size', self.batch_size)
                self.max_depth = metadata.get('max_depth', self.max_depth)
                self.min_samples_leaf = metadata.get('min_samples_leaf', self.min_samples_leaf)
                if 'tree_configs' in metadata:
                    self.tree_configs = metadata['tree_configs']

            # Load NNs
            first_nn_path = os.path.join(model_root_folder_path, 'bc_mixed_ensemble_nn_0.chkpt')
            nn_loaded = 0
            if self.n_nn > 0 and os.path.exists(first_nn_path):
                for i in range(self.n_nn):
                    net_path = os.path.join(model_root_folder_path, f'bc_mixed_ensemble_nn_{i}.chkpt')
                    if os.path.exists(net_path):
                        self.networks[i].load_checkpoint(net_path)
                        nn_loaded += 1
                    else:
                        print(f"Warning: NN {i} not found at {net_path}")

            # Load DTs
            first_dt_path = os.path.join(model_root_folder_path, 'bc_mixed_ensemble_dt_0.joblib')
            dt_loaded = 0
            if self.n_dt > 0 and os.path.exists(first_dt_path):
                loaded_trees = []
                for i in range(self.n_dt):
                    dt_path = os.path.join(model_root_folder_path, f'bc_mixed_ensemble_dt_{i}.joblib')
                    if os.path.exists(dt_path):
                        loaded_trees.append(joblib.load(dt_path))
                        dt_loaded += 1
                    else:
                        print(f"Warning: DT {i} not found at {dt_path}")
                if dt_loaded == self.n_dt:
                    self.trees = loaded_trees

            if nn_loaded == self.n_nn and dt_loaded == self.n_dt:
                self.is_trained = True
                print(f"Successfully loaded Mixed Ensemble ({self.n_nn} NNs + {self.n_dt} DTs) "
                      f"from {model_root_folder_path}")
            elif nn_loaded > 0 or dt_loaded > 0:
                print(f"Warning: Partially loaded ({nn_loaded}/{self.n_nn} NNs, "
                      f"{dt_loaded}/{self.n_dt} DTs)")
            else:
                print(f"No saved mixed ensemble found at {model_root_folder_path}, using fresh models")

        except Exception as msg:
            print(f"Error loading Mixed Ensemble agent: {msg}")

    def get_hyperparameters(self):
        """Get agent hyperparameters."""
        return {
            'state_dimension': self.state_dimension,
            'number_of_neurons': self.number_of_neurons,
            'number_of_actions': self.number_of_actions,
            'n_estimators': self.n_estimators,
            'n_nn': self.n_nn,
            'n_dt': self.n_dt,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf
        }

    def get_agreement(self, state: np.ndarray) -> float:
        """Get the proportion of ensemble members that agree on the prediction.

        Args:
            state: Input state

        Returns:
            Float between 0 and 1 indicating agreement level
        """
        if not self.is_trained:
            return 0.0

        actions = self.get_ensemble_actions(state)
        unique, counts = np.unique(actions, return_counts=True)
        max_count = counts.max()

        return max_count / self.n_estimators
