import mlflow
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from common.agents.stochastic_agent import StochasticAgent
from common.utilities.helper import *
from collections import OrderedDict
import numpy as np


class Policy(nn.Module):
    def __init__(self, state_dimension: int, number_of_neurons: list, number_of_actions: int, lr: float):
        super(Policy, self).__init__()

        layers = OrderedDict()
        previous_neurons = state_dimension
        for i in range(len(number_of_neurons)):
            layers[str(i)] = nn.Linear(previous_neurons, number_of_neurons[i])
            previous_neurons = number_of_neurons[i]
        self.layers = nn.Sequential(layers)
        self.output = nn.Linear(previous_neurons, number_of_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(DEVICE)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).to(DEVICE)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = state
        for layer in self.layers:
            x = F.relu(layer(x))
        return F.softmax(self.output(x), dim=-1)

    def act(self, state, deploy=False):
        probs = self.forward(state)
        dist = Categorical(probs)
        # For stochastic policy, always sample from distribution (even during deploy)
        # This ensures the policy remains stochastic during evaluation and model checking
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def save_checkpoint(self, file_name: str):
        torch.save(self.state_dict(), file_name)

    def load_checkpoint(self, file_name: str):
        self.load_state_dict(torch.load(file_name))


class StochasticPPOAgent(StochasticAgent):
    """
    Stochastic Proximal Policy Optimization (PPO) Agent.

    This is identical to PPOAgent but extends StochasticAgent and implements
    action_probability_distribution for model checking with stochastic policies.

    This implementation works with both rewards (positive values) and penalties (negative values).

    How it handles penalties:
    --------------------------
    PPO uses advantage normalization which makes the algorithm agnostic to reward sign:

    1. Raw returns are computed: R_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
       - With rewards: returns might be [10, 8, 5, 2]
       - With penalties: returns might be [-10, -8, -5, -2]

    2. Normalization transforms returns to advantages:
       A = (R - mean(R)) / std(R)
       - This centers the distribution around 0
       - Better-than-average trajectories get positive advantages
       - Worse-than-average trajectories get negative advantages

    3. PPO objective: maximize E[min(r*A, clip(r,1-ε,1+ε)*A)]
       - Positive advantage → increase action probability
       - Negative advantage → decrease action probability

    Example with penalties [-10, -8, -5, -2]:
       mean = -6.25, std ≈ 3.3
       normalized = [-1.13, -0.53, 0.38, 1.28]
       → Actions leading to -2 (least bad) get positive advantage
       → Actions leading to -10 (worst) get negative advantage

    The key insight: normalization converts "less penalty" into "positive advantage",
    so the agent learns to minimize penalties just as effectively as maximizing rewards.
    """

    # Minimum number of samples required before running PPO update
    MIN_SAMPLES_FOR_UPDATE = 2

    def __init__(self, state_dimension: int, number_of_neurons: list, number_of_actions: int,
                 gamma: float = 0.99, lr: float = 0.0003, clip_epsilon: float = 0.2,
                 ppo_epochs: int = 4, batch_size: int = 64, entropy_coef: float = 0.01):
        super().__init__()
        self.policy = Policy(state_dimension, number_of_neurons, number_of_actions, lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.skip_ppo_updates = False  # Set to True after BC to use BC policy only

        # Shielded training mode: when True, skip advantage normalization
        # This is needed because with shielding, all trajectories get similar rewards,
        # and normalization would make all advantages ~0, removing the learning signal.
        # With raw returns as advantages, the agent still reinforces optimal actions.
        self.shielded_training = False

        # Episode buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

        # Temporary storage for shield correction
        self._last_state = None
        self._last_log_prob = None
        self._last_action = None

    def select_action(self, state: np.ndarray, deploy: bool = False) -> int:
        action, log_prob = self.policy.act(state, deploy)
        if not deploy:
            self.states.append(state)
            self._last_state = state
            self._last_log_prob = log_prob
            self._last_action = action
        return action

    def action_probability_distribution(self, state: np.ndarray) -> np.ndarray:
        """
        Get the action probability distribution of the agent for a given state.

        Args:
            state (np.ndarray): The state of the environment.

        Returns:
            np.ndarray: The action probability distribution of the agent.
                       Array of shape (num_actions,) where each element is P(action|state).
        """
        with torch.no_grad():
            probs = self.policy.forward(state)
            # Convert to numpy and squeeze to 1D array
            return probs.cpu().numpy().squeeze()

    def store_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool):
        """
        Store experience for PPO update.

        Note: The 'state' parameter is kept for interface compatibility but we use
        self._last_state internally to ensure consistency with the action selection.
        If you need to use the passed state, ensure it matches what was used in select_action.
        """
        # If action was corrected by shield, recompute log_prob for the corrected action
        if action != self._last_action:
            with torch.no_grad():
                probs = self.policy.forward(self._last_state)
                dist = Categorical(probs)
                corrected_log_prob = dist.log_prob(torch.tensor([action]).to(DEVICE))
            # Use .item() directly - it handles any tensor shape
            self.log_probs.append(corrected_log_prob.item())
        else:
            # FIX: Use .item() directly instead of .squeeze(0).item()
            # This handles both 0-dim tensors and 1-dim tensors safely
            self.log_probs.append(self._last_log_prob.item())

        self.actions.append(action)
        self.rewards.append(reward)

    def _normalize_returns(self, returns: torch.Tensor) -> torch.Tensor:
        """
        Normalize returns to compute advantages.

        This method handles edge cases that would otherwise cause NaN:
        1. Single sample: std() returns NaN with unbiased=True (divides by n-1=0)
        2. Identical returns: std() returns 0, causing division by zero

        The normalization makes PPO work with both rewards and penalties:
        - Better-than-average returns → positive advantages → increase probability
        - Worse-than-average returns → negative advantages → decrease probability

        Args:
            returns: Tensor of discounted returns

        Returns:
            Normalized advantages tensor
        """
        n = len(returns)

        # Edge case 1: Single sample - can't compute meaningful statistics
        # Just center around 0 (or return as-is since there's nothing to compare)
        if n == 1:
            return returns - returns.mean()

        # Compute mean and std
        mean = returns.mean()
        # Use unbiased=False to avoid NaN with small samples
        # For PPO, biased std is fine since we just need relative scaling
        std = returns.std(unbiased=False)

        # Edge case 2: All returns are identical (std ≈ 0)
        # This can happen if all rewards are the same (e.g., all -1 penalties)
        # Just center the returns; all actions are equally good/bad
        if std < 1e-8:
            return returns - mean

        # Standard normalization
        return (returns - mean) / (std + 1e-8)

    def episodic_learn(self):
        """
        Perform PPO update at the end of an episode.

        This method:
        1. Computes discounted returns from rewards/penalties
        2. Normalizes returns to get advantages (works for both + and - rewards)
        3. Performs multiple epochs of PPO clipped objective optimization
        4. Clears episode buffers
        """
        # Skip if not enough samples
        if len(self.states) < self.MIN_SAMPLES_FOR_UPDATE:
            self._clear_buffers()
            return

        # Skip PPO updates if flag is set (e.g., after BC pre-training)
        if self.skip_ppo_updates:
            self._clear_buffers()
            return

        # Compute discounted returns
        # Works identically for rewards (positive) and penalties (negative)
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)

        # Compute advantages
        if self.shielded_training:
            # In shielded training mode, use raw returns as advantages (no normalization).
            # With shielding, all trajectories are optimal → similar returns → normalization
            # would make all advantages ~0. Raw returns preserve the learning signal.
            advantages = returns
        else:
            # Standard PPO: normalize returns to get advantages
            advantages = self._normalize_returns(returns)

        # Convert to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(self.actions, dtype=torch.long).to(DEVICE)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32).to(DEVICE)

        # PPO update with mini-batches
        dataset_size = len(states)

        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_idx = indices[start:end]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]

                # Get new action probabilities
                probs = self.policy.forward(batch_states)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                # ratio = π_new(a|s) / π_old(a|s)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                # Clipped surrogate objectives
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages

                # Take minimum for conservative update
                # This works correctly for both positive and negative advantages:
                # - Positive advantage: min clips large ratio increases
                # - Negative advantage: min clips large ratio decreases
                policy_loss = -torch.min(surr1, surr2).mean()

                # Add entropy bonus to encourage exploration
                # Subtracting entropy (negative of negative) increases entropy
                loss = policy_loss - self.entropy_coef * entropy

                # Check for NaN before backward pass
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected. Skipping batch update.")
                    print(f"  - advantages stats: mean={batch_advantages.mean():.4f}, "
                          f"std={batch_advantages.std():.4f}, min={batch_advantages.min():.4f}, "
                          f"max={batch_advantages.max():.4f}")
                    continue

                self.policy.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy.optimizer.step()

        # Clear buffers
        self._clear_buffers()

    def _clear_buffers(self):
        """Clear all episode buffers."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []

    def step_learn(self):
        """Called after each step. PPO uses episodic learning, so this is a no-op."""
        pass

    def behavioral_cloning(self, env, data: dict, epochs: int = 100, accuracy_threshold: float = 100.0):
        """
        Pre-train the PPO policy using behavioral cloning (supervised learning).

        Args:
            env: The environment (not used but required for interface compatibility)
            data: Dict with 'X_train', 'y_train', optionally 'X_test', 'y_test'
            epochs: Number of training epochs

        Returns:
            Tuple of (best_epoch, train_accuracy, test_accuracy, train_loss, test_loss)
        """
        X_train = data.get('X_train')
        y_train = data.get('y_train')
        X_test = data.get('X_test')
        y_test = data.get('y_test')

        if X_train is None or y_train is None or len(X_train) == 0 or len(y_train) == 0:
            print("Warning: No training data available for behavioral cloning")
            return None, None, None, None, None

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(DEVICE)

        print(f"Training data: {len(X_train)} samples")

        has_test_data = X_test is not None and y_test is not None and len(X_test) > 0
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
        unique_classes, class_counts = torch.unique(y_train_tensor, return_counts=True)
        num_classes = self.policy.output.out_features

        class_weights = torch.ones(num_classes, dtype=torch.float32)
        for cls, count in zip(unique_classes, class_counts):
            class_weights[cls] = num_samples / (num_classes * count.item())
        class_weights = class_weights.to(DEVICE)

        print(f"\nBehavioral Cloning Pre-Training for Stochastic PPO")
        print(f"Class distribution in training data:")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Action {cls.item()}: {count.item()} samples (weight: {class_weights[cls].item():.4f})")

        # Create weighted cross-entropy loss
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Save best model during training
        temp_model_dir = 'tmp_bc_stochastic_ppo_best_model'
        os.makedirs(temp_model_dir, exist_ok=True)
        best_model_checkpoint = os.path.join(temp_model_dir, 'best_stochastic_ppo_policy.chkpt')

        for epoch in range(epochs):
            self.policy.train()

            # Shuffle training data
            indices = torch.randperm(num_samples)
            X_train_shuffled = X_train_tensor[indices]
            y_train_shuffled = y_train_tensor[indices]

            total_train_loss = 0.0
            correct_train = 0
            total_train = 0
            num_batches = 0

            # Mini-batch training
            for i in range(0, num_samples, self.batch_size):
                batch_X = X_train_shuffled[i:i + self.batch_size]
                batch_y = y_train_shuffled[i:i + self.batch_size]

                self.policy.optimizer.zero_grad()

                # Get logits (before softmax) for cross-entropy loss
                x = batch_X
                for layer in self.policy.layers:
                    x = F.relu(layer(x))
                logits = self.policy.output(x)

                loss = loss_fn(logits, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy.optimizer.step()

                total_train_loss += loss.item()
                num_batches += 1

                # Calculate batch accuracy
                _, predicted = torch.max(logits.data, 1)
                total_train += batch_y.size(0)
                correct_train += (predicted == batch_y).sum().item()

            train_loss = total_train_loss / max(1, num_batches)
            train_accuracy = 100 * correct_train / max(1, total_train)

            # Evaluate on test set if available
            test_accuracy = None
            test_loss = None
            if has_test_data:
                self.policy.eval()
                with torch.no_grad():
                    x = X_test_tensor
                    for layer in self.policy.layers:
                        x = F.relu(layer(x))
                    test_logits = self.policy.output(x)
                    test_loss_val = loss_fn(test_logits, y_test_tensor)
                    test_loss = test_loss_val.item()

                    _, test_predicted = torch.max(test_logits.data, 1)
                    test_accuracy = 100 * (test_predicted == y_test_tensor).sum().item() / y_test_tensor.size(0)

                if test_accuracy > best_test_accuracy:
                    best_test_accuracy = test_accuracy
                    best_test_loss = test_loss
                    best_train_accuracy = train_accuracy
                    best_train_loss = train_loss
                    best_epoch = epoch
                    self.policy.save_checkpoint(best_model_checkpoint)
            else:
                if train_accuracy > best_train_accuracy:
                    best_train_accuracy = train_accuracy
                    best_train_loss = train_loss
                    best_epoch = epoch
                    self.policy.save_checkpoint(best_model_checkpoint)

            print(f"Epoch {epoch + 1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_accuracy:.2f}%", end="")
            if has_test_data:
                print(f", Test Loss={test_loss:.4f}, Test Acc={test_accuracy:.2f}%")
            else:
                print()

        # Load best model after training
        if os.path.exists(best_model_checkpoint):
            self.policy.load_checkpoint(best_model_checkpoint)
            print(f"\n{'='*70}")
            print(f"Stochastic PPO Behavioral Cloning Pre-Training Complete!")
            print(f"{'='*70}")
            print(f"Best Model (saved at epoch {best_epoch + 1}):")
            print(f"  Train Accuracy: {best_train_accuracy:.2f}%")
            print(f"  Train Loss:     {best_train_loss:.4f}")
            if has_test_data:
                print(f"  Test Accuracy:  {best_test_accuracy:.2f}%")
                print(f"  Test Loss:      {best_test_loss:.4f}")
            print(f"{'='*70}\n")

        # Clean up
        if os.path.exists(temp_model_dir):
            shutil.rmtree(temp_model_dir)

        # Reset optimizer with lower learning rate for PPO fine-tuning
        # This prevents PPO from immediately destroying the BC-learned policy
        fine_tune_lr = self.policy.optimizer.param_groups[0]['lr'] * 0.1
        self.policy.optimizer = optim.Adam(self.policy.parameters(), lr=fine_tune_lr)

        # Reduce PPO aggressiveness for fine-tuning
        self.ppo_epochs = max(1, self.ppo_epochs // 2)  # Fewer update passes
        self.entropy_coef = self.entropy_coef * 0.1     # Less exploration needed

        print(f"Stochastic PPO fine-tuning settings:")
        print(f"  Learning rate: {fine_tune_lr}")
        print(f"  PPO epochs: {self.ppo_epochs}")
        print(f"  Entropy coef: {self.entropy_coef}")

        # Continue with PPO fine-tuning after BC
        # self.skip_ppo_updates = True  # Uncomment to disable PPO updates

        return best_epoch, best_train_accuracy, best_test_accuracy, best_train_loss, best_test_loss

    def save(self, artifact_path='model'):
        try:
            os.mkdir('tmp_model')
        except FileExistsError:
            pass
        self.policy.save_checkpoint('tmp_model/policy.chkpt')
        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)
        shutil.rmtree('tmp_model')

    def load(self, model_root_folder_path: str):
        try:
            self.policy.load_checkpoint(os.path.join(model_root_folder_path, 'policy.chkpt'))
        except Exception as msg:
            print(msg)

    def get_hyperparameters(self):
        """Get the RL agent hyperparameters"""
        return {
            'gamma': self.gamma,
            'lr': self.policy.optimizer.param_groups[0]['lr'],
            'clip_epsilon': self.clip_epsilon,
            'ppo_epochs': self.ppo_epochs,
            'batch_size': self.batch_size,
            'entropy_coef': self.entropy_coef
        }
