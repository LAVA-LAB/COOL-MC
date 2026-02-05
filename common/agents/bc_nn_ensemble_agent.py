"""
Behavioral Cloning Agent using Neural Network Ensemble with Majority Voting.

This agent trains multiple neural networks on the SAME data with different
random initializations and uses majority voting for prediction.
"""
import os
import shutil
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List
from collections import OrderedDict
from scipy import stats
from common.agents.agent import Agent
from common.utilities.helper import DEVICE


class EnsembleNetwork(nn.Module):
    """Single neural network for the ensemble."""

    def __init__(self, state_dimension: int, number_of_neurons: List[int],
                 number_of_actions: int, lr: float, seed: int):
        super(EnsembleNetwork, self).__init__()

        # Set seed for reproducibility but different for each network
        torch.manual_seed(seed)

        layers = OrderedDict()
        previous_neurons = state_dimension
        for i in range(len(number_of_neurons) + 1):
            if i == len(number_of_neurons):
                layers[str(i)] = nn.Linear(previous_neurons, number_of_actions)
            else:
                layers[str(i)] = nn.Linear(previous_neurons, number_of_neurons[i])
                previous_neurons = number_of_neurons[i]
        self.layers = nn.Sequential(layers)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.DEVICE = DEVICE
        self.to(self.DEVICE)

    def forward(self, state):
        try:
            x = state
            for i in range(len(self.layers)):
                if i == (len(self.layers) - 1):
                    x = self.layers[i](x)
                else:
                    x = F.relu(self.layers[i](x))
            return x
        except:
            state = torch.tensor(state).float().to(self.DEVICE)
            x = state
            for i in range(len(self.layers)):
                if i == (len(self.layers) - 1):
                    x = self.layers[i](x)
                else:
                    x = F.relu(self.layers[i](x))
            return x

    def save_checkpoint(self, file_name: str):
        torch.save(self.state_dict(), file_name)

    def load_checkpoint(self, file_name: str):
        self.load_state_dict(torch.load(file_name, map_location=self.DEVICE))


class BCNNEnsembleAgent(Agent):
    """
    Behavioral Cloning agent using an ensemble of Neural Networks with majority voting.

    Each network is initialized with a different random seed, providing diversity
    in the ensemble. The final prediction uses majority voting across all networks.
    """

    def __init__(self, state_dimension: int, number_of_neurons: List[int],
                 number_of_actions: int, n_estimators: int = 5,
                 learning_rate: float = 0.001, batch_size: int = 64):
        """Initialize BC Neural Network Ensemble Agent.

        Args:
            state_dimension (int): State dimension
            number_of_neurons (List[int]): Neurons per hidden layer
            number_of_actions (int): Number of actions
            n_estimators (int, optional): Number of networks in ensemble. Defaults to 5.
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            batch_size (int, optional): Batch size for training. Defaults to 64.
        """
        self.state_dimension = state_dimension
        self.number_of_neurons = number_of_neurons
        self.number_of_actions = number_of_actions
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Create ensemble of networks with different random seeds
        self.networks = [
            EnsembleNetwork(
                state_dimension, number_of_neurons, number_of_actions,
                learning_rate, seed=42 + i
            )
            for i in range(n_estimators)
        ]
        self.is_trained = False

    def behavioral_cloning(self, env, data: dict, epochs: int = 100, accuracy_threshold: float = 100.0):
        """
        Train all neural networks in the ensemble on behavioral cloning data.

        All networks are trained together epoch by epoch, allowing early stopping
        when every individual network reaches the accuracy threshold.

        Args:
            env: Environment
            data: Dict with 'X_train', 'y_train', 'X_test', 'y_test'
            epochs: Number of training epochs
            accuracy_threshold: Stop early when all individual networks >= this value (in percent)

        Returns:
            Tuple of (epoch, train_acc, test_acc, train_loss, test_loss)
        """
        X_train = data.get('X_train')
        y_train = data.get('y_train')
        X_test = data.get('X_test')
        y_test = data.get('y_test')

        if X_train is None or y_train is None:
            return None, None, None, None, None

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(DEVICE)

        has_test_data = X_test is not None and y_test is not None
        if has_test_data:
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
            y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

        num_samples = X_train_tensor.shape[0]

        # Compute class weights
        unique_classes, class_counts = torch.unique(y_train_tensor, return_counts=True)

        print(f"\nClass distribution in training data:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Action {cls.item()}: {count.item()} samples")

        class_weights = torch.ones(self.number_of_actions, dtype=torch.float32)
        for cls, count in zip(unique_classes, class_counts):
            class_weights[cls] = num_samples / (self.number_of_actions * count.item())
        class_weights = class_weights.to(DEVICE)

        weighted_loss = nn.CrossEntropyLoss(weight=class_weights)

        print(f"\nTraining Neural Network Ensemble (n_estimators={self.n_estimators}, "
              f"architecture={self.number_of_neurons}, lr={self.learning_rate})...")
        print(f"Early stopping when all individual networks >= {accuracy_threshold}%")

        # Track best accuracy and checkpoints per network
        best_accuracies = [0.0] * self.n_estimators
        temp_checkpoints = [f'tmp_nn_ensemble_best_{i}.chkpt' for i in range(self.n_estimators)]

        # Train all networks together, epoch by epoch
        final_epoch = epochs - 1
        for epoch in range(epochs):
            net_accs = []
            for net_idx, network in enumerate(self.networks):
                network.train()

                # Shuffle training data (each network gets its own shuffle)
                indices = torch.randperm(num_samples)
                X_shuffled = X_train_tensor[indices]
                y_shuffled = y_train_tensor[indices]

                total_loss = 0.0
                correct = 0
                total = 0

                # Mini-batch training
                for i in range(0, num_samples, self.batch_size):
                    batch_X = X_shuffled[i:i + self.batch_size]
                    batch_y = y_shuffled[i:i + self.batch_size]

                    network.optimizer.zero_grad()
                    outputs = network.forward(batch_X)
                    loss = weighted_loss(outputs, batch_y)
                    loss.backward()
                    network.optimizer.step()

                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

                net_accuracy = 100 * correct / total
                net_accs.append(net_accuracy)

                # Save best model per network
                if net_accuracy > best_accuracies[net_idx]:
                    best_accuracies[net_idx] = net_accuracy
                    network.save_checkpoint(temp_checkpoints[net_idx])

            # Progress logging
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                avg_acc = np.mean(net_accs)
                print(f"  Epoch {epoch + 1}/{epochs}: Avg Net Acc={avg_acc:.2f}% "
                      f"(individual: {', '.join(f'{a:.1f}%' for a in net_accs)})")

            # Check if ALL individual networks reached the accuracy threshold
            if all(acc >= accuracy_threshold for acc in net_accs):
                print(f"\n  Early stopping at epoch {epoch + 1}/{epochs}: "
                      f"all networks >= {accuracy_threshold}% "
                      f"(individual: {', '.join(f'{a:.1f}%' for a in net_accs)})")
                final_epoch = epoch
                break

        # Load best checkpoint for each network
        for i in range(self.n_estimators):
            if os.path.exists(temp_checkpoints[i]):
                self.networks[i].load_checkpoint(temp_checkpoints[i])
                os.remove(temp_checkpoints[i])
            print(f"  Network {i + 1} best accuracy: {best_accuracies[i]:.2f}%")

        self.is_trained = True

        # Compute ensemble accuracy using majority voting
        y_train_pred = self._predict_majority_vote(X_train)
        train_accuracy = 100 * np.mean(y_train_pred == y_train)

        test_accuracy = None
        if has_test_data:
            y_test_pred = self._predict_majority_vote(X_test)
            test_accuracy = 100 * np.mean(y_test_pred == y_test)

        print(f"\n{'='*70}")
        print(f"Neural Network Ensemble Training Complete!")
        print(f"{'='*70}")
        print(f"Ensemble Statistics:")
        print(f"  Number of networks: {self.n_estimators}")
        print(f"  Architecture:       {self.number_of_neurons}")
        print(f"  Epochs completed:   {final_epoch + 1}/{epochs}")
        print(f"  Ensemble Train Accuracy: {train_accuracy:.2f}%")
        if test_accuracy is not None:
            print(f"  Ensemble Test Accuracy:  {test_accuracy:.2f}%")
        print(f"{'='*70}\n")

        return final_epoch, train_accuracy, test_accuracy, 0.0, 0.0

    def _predict_majority_vote(self, X: np.ndarray) -> np.ndarray:
        """Get predictions using majority voting across all networks.

        Args:
            X: Input features (n_samples, n_features)

        Returns:
            Array of predicted class labels
        """
        X_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)

        all_predictions = []
        for network in self.networks:
            network.eval()
            with torch.no_grad():
                outputs = network.forward(X_tensor)
                _, predicted = torch.max(outputs, 1)
                all_predictions.append(predicted.cpu().numpy())

        predictions = np.array(all_predictions)  # (n_networks, n_samples)
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
        """Get individual actions from all neural networks in the ensemble.

        Args:
            state (np.ndarray): Current state

        Returns:
            list[int]: List of actions, one from each network in the ensemble

        Raises:
            RuntimeError: If the ensemble has not been trained yet.
        """
        if not self.is_trained:
            raise RuntimeError("Ensemble has not been trained yet.")

        state_2d = state.reshape(1, -1) if state.ndim == 1 else state
        state_tensor = torch.tensor(state_2d, dtype=torch.float32).to(DEVICE)

        actions = []
        for network in self.networks:
            network.eval()
            with torch.no_grad():
                outputs = network.forward(state_tensor)
                _, predicted = torch.max(outputs, 1)
                actions.append(int(predicted.cpu().item()))

        return actions

    def get_raw_outputs(self, state: np.ndarray) -> np.ndarray:
        """Get vote proportions for each action (soft voting).

        Args:
            state (np.ndarray): Current state

        Returns:
            np.ndarray: Proportion of networks voting for each action
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
        """Save the ensemble to MLFlow artifacts."""
        try:
            os.makedirs('tmp_model', exist_ok=True)
        except Exception:
            pass

        # Save all networks
        for i, network in enumerate(self.networks):
            network.save_checkpoint(f'tmp_model/bc_nn_ensemble_net_{i}.chkpt')

        # Save metadata
        metadata = {
            'state_dimension': self.state_dimension,
            'number_of_neurons': self.number_of_neurons,
            'number_of_actions': self.number_of_actions,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'is_trained': self.is_trained
        }
        torch.save(metadata, 'tmp_model/bc_nn_ensemble_meta.pt')

        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)
        shutil.rmtree('tmp_model')

    def load(self, model_root_folder_path: str):
        """Load the ensemble from disk.

        Args:
            model_root_folder_path (str): Path to model folder
        """
        try:
            # Load metadata first
            meta_path = os.path.join(model_root_folder_path, 'bc_nn_ensemble_meta.pt')
            if os.path.exists(meta_path):
                metadata = torch.load(meta_path, map_location=DEVICE)
                self.n_estimators = metadata.get('n_estimators', self.n_estimators)
                self.learning_rate = metadata.get('learning_rate', self.learning_rate)
                self.batch_size = metadata.get('batch_size', self.batch_size)

            # Check if any network files exist before clearing
            first_net_path = os.path.join(model_root_folder_path, 'bc_nn_ensemble_net_0.chkpt')
            if not os.path.exists(first_net_path):
                print(f"No saved ensemble networks found at {model_root_folder_path}, using fresh networks")
                return

            # Load all networks
            loaded_count = 0
            for i in range(self.n_estimators):
                net_path = os.path.join(model_root_folder_path, f'bc_nn_ensemble_net_{i}.chkpt')
                if os.path.exists(net_path):
                    self.networks[i].load_checkpoint(net_path)
                    loaded_count += 1
                else:
                    print(f"Warning: Network {i} not found at {net_path}")

            if loaded_count == self.n_estimators:
                self.is_trained = True
                print(f"Successfully loaded Neural Network Ensemble ({self.n_estimators} networks) from {model_root_folder_path}")
            else:
                print(f"Warning: Only loaded {loaded_count}/{self.n_estimators} networks, keeping fresh networks")

        except Exception as msg:
            print(f"Error loading Neural Network Ensemble agent: {msg}")

    def get_hyperparameters(self):
        """Get agent hyperparameters."""
        return {
            'state_dimension': self.state_dimension,
            'number_of_neurons': self.number_of_neurons,
            'number_of_actions': self.number_of_actions,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }

    def get_network_agreement(self, state: np.ndarray) -> float:
        """Get the proportion of networks that agree on the prediction.

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
