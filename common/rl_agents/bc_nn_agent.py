import mlflow
import os
import shutil
from typing import List
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.rl_agents.agent import Agent
from common.utilities.helper import *
from collections import OrderedDict
import torch
import numpy as np


class BCNetwork(nn.Module):
    def __init__(self, state_dimension: int, number_of_neurons: List[int], number_of_actions: int, lr: float):
        """
        Neural network for behavioral cloning.

        Args:
            state_dimension (int): The dimension of the state
            number_of_neurons (List[int]): List of neurons. Each element is a new layer.
            number_of_actions (int): Number of actions.
            lr (float): Learning rate.
        """
        super(BCNetwork, self).__init__()

        layers = OrderedDict()
        previous_neurons = state_dimension
        for i in range(len(number_of_neurons) + 1):
            if i == len(number_of_neurons):
                layers[str(i)] = torch.nn.Linear(previous_neurons, number_of_actions)
            else:
                layers[str(i)] = torch.nn.Linear(previous_neurons, number_of_neurons[i])
                previous_neurons = number_of_neurons[i]
        self.layers = torch.nn.Sequential(layers)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        self.DEVICE = DEVICE
        self.to(self.DEVICE)

    def forward(self, state: np.array) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state (np.array): State

        Returns:
            torch.Tensor: Action logits
        """
        try:
            x = state
            for i in range(len(self.layers)):
                if i == (len(self.layers) - 1):
                    x = self.layers[i](x)
                else:
                    x = F.relu(self.layers[i](x))
            return x
        except:
            state = torch.tensor(state).float().to(DEVICE)
            x = state
            for i in range(len(self.layers)):
                if i == (len(self.layers) - 1):
                    x = self.layers[i](x)
                else:
                    x = F.relu(self.layers[i](x))
            return x

    def save_checkpoint(self, file_name: str):
        """Save model.

        Args:
            file_name (str): File name
        """
        torch.save(self.state_dict(), file_name)

    def load_checkpoint(self, file_name: str):
        """Load model

        Args:
            file_name (str): File name
        """
        self.load_state_dict(torch.load(file_name))



class BCNNAgent(Agent):

    def __init__(self, state_dimension: int, number_of_neurons, number_of_actions: List[int], learning_rate: float = 0.001, batch_size: int = 64):
        """Initialize Behavioral Cloning Neural Network Agent

        Args:
            state_dimension (int): State Dimension
            number_of_neurons (List[int]): Each element is a new layer and defines the number of neurons in the current layer
            number_of_actions (int): Number of Actions
            learning_rate (float, optional): Learning rate. Defaults to 0.001.
            batch_size (int, optional): Batch Size. Defaults to 64.
        """
        self.number_of_actions = number_of_actions
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.state_dimension = state_dimension
        self.number_of_neurons = number_of_neurons

        self.network = BCNetwork(state_dimension, number_of_neurons, number_of_actions, learning_rate)
        self.best_model_path = None

    def behavioral_cloning(self, data: dict, epochs: int = 100) -> tuple[int | None, float | None, float | None, float | None, float | None]:
        """
        Perform supervised training on a behavioral cloning dataset.
        Uses class weighting to handle imbalanced action labels.

        Args:
            data: The behavioral cloning dataset with keys 'X_train', 'y_train', 'X_test', 'y_test'.
            epochs: Number of training epochs.

        Returns:
            A tuple of (training_epoch, train_accuracy, test_accuracy, train_loss, test_loss).
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
        best_test_accuracy = 0.0
        best_train_accuracy = 0.0
        best_epoch = 0
        best_train_loss = float('inf')
        best_test_loss = float('inf')

        # Compute class weights to handle imbalanced data
        # Weight = total_samples / (num_classes * samples_per_class)
        unique_classes, class_counts = torch.unique(y_train_tensor, return_counts=True)
        num_classes = self.number_of_actions

        # Initialize weights for all possible action classes
        class_weights = torch.ones(num_classes, dtype=torch.float32)

        # Compute inverse frequency weights for classes present in training data
        for cls, count in zip(unique_classes, class_counts):
            class_weights[cls] = num_samples / (num_classes * count.item())

        class_weights = class_weights.to(DEVICE)

        # Print class distribution for debugging
        print(f"\nClass distribution in training data:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Action {cls.item()}: {count.item()} samples (weight: {class_weights[cls].item():.4f})")

        # Create weighted loss function
        weighted_loss = nn.CrossEntropyLoss(weight=class_weights)

        # Create temporary directory for best model during training
        temp_model_dir = 'tmp_bc_best_model'
        os.makedirs(temp_model_dir, exist_ok=True)
        best_model_checkpoint = os.path.join(temp_model_dir, 'best_bc_model.chkpt')

        for epoch in range(epochs):
            self.network.train()

            # Shuffle training data
            indices = torch.randperm(num_samples)
            X_train_shuffled = X_train_tensor[indices]
            y_train_shuffled = y_train_tensor[indices]

            total_train_loss = 0.0
            correct_train = 0
            total_train = 0

            # Mini-batch training
            for i in range(0, num_samples, self.batch_size):
                batch_X = X_train_shuffled[i:i + self.batch_size]
                batch_y = y_train_shuffled[i:i + self.batch_size]

                self.network.optimizer.zero_grad()
                outputs = self.network.forward(batch_X)
                loss = weighted_loss(outputs, batch_y)  # Use weighted loss
                loss.backward()
                self.network.optimizer.step()

                total_train_loss += loss.item()

                # Calculate batch accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()

            train_loss = total_train_loss / (num_samples / self.batch_size)
            train_accuracy = 100 * correct_train / total_train

            # Evaluate on test set if available
            test_accuracy = None
            test_loss = None
            if has_test_data:
                self.network.eval()
                with torch.no_grad():
                    test_outputs = self.network.forward(X_test_tensor)
                    test_loss_val = weighted_loss(test_outputs, y_test_tensor)  # Use weighted loss
                    test_loss = test_loss_val.item()

                    _, test_predicted = torch.max(test_outputs.data, 1)
                    test_accuracy = 100 * (test_predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)

                # Save best model based on test accuracy
                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    best_test_loss = test_loss
                    best_train_accuracy = train_accuracy
                    best_train_loss = train_loss
                    best_epoch = epoch
                    self.network.save_checkpoint(best_model_checkpoint)
                    self.best_model_path = best_model_checkpoint
            else:
                # Save best model based on training accuracy when no test data
                if train_accuracy > best_train_accuracy:
                    best_train_accuracy = train_accuracy
                    best_train_loss = train_loss
                    best_epoch = epoch
                    self.network.save_checkpoint(best_model_checkpoint)
                    self.best_model_path = best_model_checkpoint

            # Print progress every epoch
            print(f"Epoch {epoch + 1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.2f}%", end="")
            if has_test_data:
                print(f", Test Loss={test_loss:.4f}, Test Acc={test_accuracy:.2f}%")
            else:
                print()

        # Load best model after training
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.network.load_checkpoint(self.best_model_path)
            print(f"\n{'='*70}")
            print(f"Behavioral Cloning Training Complete!")
            print(f"{'='*70}")
            print(f"Best Model (saved at epoch {best_epoch + 1}):")
            if has_test_data:
                print(f"  Train Accuracy: {best_train_accuracy:.2f}%")
                print(f"  Train Loss:     {best_train_loss:.4f}")
                print(f"  Test Accuracy:  {best_test_accuracy:.2f}%")
                print(f"  Test Loss:      {best_test_loss:.4f}")
            else:
                print(f"  Train Accuracy: {best_train_accuracy:.2f}%")
                print(f"  Train Loss:     {best_train_loss:.4f}")
            print(f"{'='*70}\n")

        # Clean up temporary directory used during training
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)

        return best_epoch, best_train_accuracy, best_test_accuracy, best_train_loss, best_test_loss

    def select_action(self, state: np.ndarray, deploy: bool = False) -> int:
        """Select action based on the current state using the trained BC policy.

        Args:
            state (np.ndarray): Current state
            deploy (bool, optional): If true, always use greedy action selection. Defaults to False.

        Returns:
            int: action
        """
        self.network.eval()
        with torch.no_grad():
            action_logits = self.network.forward(state)
            action = int(torch.argmax(action_logits).item())
        return action

    def save(self, artifact_path='model'):
        """
        Saves the agent onto the MLFlow Server.
        """
        try:
            os.mkdir('tmp_model')
        except Exception as msg:
            pass

        self.network.save_checkpoint('tmp_model/bc_network.chkpt')
        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)
        shutil.rmtree('tmp_model')

    def load(self, model_root_folder_path: str):
        """Loads the Agent from the MLFlow server or local folder.

        Args:
            model_root_folder_path (str): Model root folder path.
        """
        try:
            checkpoint_path = os.path.join(model_root_folder_path, 'bc_network.chkpt')
            if os.path.exists(checkpoint_path):
                self.network.load_checkpoint(checkpoint_path)
                print(f"Successfully loaded BC agent from {checkpoint_path}")
            else:
                print(f"Warning: Checkpoint file not found at {checkpoint_path}")
        except Exception as msg:
            print(f"Error loading BC agent: {msg}")

    def get_hyperparameters(self):
        """
        Get the RL agent hyperparameters
        """
        return {
            'state_dimension': self.state_dimension,
            'number_of_neurons': self.number_of_neurons,
            'number_of_actions': self.number_of_actions,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }


