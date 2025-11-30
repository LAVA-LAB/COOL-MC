"""
Classical REINFORCE Agent with Neural Network Policy

This agent implements the classical REINFORCE (Monte Carlo Policy Gradient) algorithm:
- Multi-layer perceptron (MLP) as the policy network
- Backpropagation for gradient calculation
- Episodic learning with Monte Carlo returns
- Baseline for comparison with quantum implementations

References:
- REINFORCE: Simple Statistical Gradient-Following Algorithms (Williams, 1992)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import mlflow
import os
import shutil
from common.rl_agents.agent import StochasticAgent
from common.utilities.helper import *
from typing import Tuple


class Policy(nn.Module):
    """
    Classical Policy Network using Multi-Layer Perceptron.

    Architecture:
    1. Input layer: state features
    2. Hidden layers: ReLU activations
    3. Output layer: action logits
    4. Softmax: convert to action probabilities
    """

    def __init__(self, s_size=4, h_size=16, a_size=2, lr=0.001):
        """
        Initialize Policy Network.

        Args:
            s_size: State dimension
            h_size: Hidden layer size
            a_size: Action dimension
            lr: Learning rate
        """
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, a_size)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        """
        Forward pass through policy network.

        Args:
            x: Input state tensor

        Returns:
            Action probabilities (softmax over actions)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        """
        Select action using current policy.

        Args:
            state: Current state

        Returns:
            Tuple of (sampled_action, log_prob, greedy_action)
        """
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action), int(torch.argmax(probs))

    def stochastic_act(self, state, prob_threshold):
        """
        Select actions stochastically with threshold filtering.

        Args:
            state: Current state
            prob_threshold: Minimum probability threshold

        Returns:
            Tuple of (valid_action_indices, probabilities)
        """
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        probs = self.forward(state).cpu()
        probs_list = probs.tolist()[0]

        # Filter actions by threshold
        action_indexes = []
        for i, prob in enumerate(probs_list):
            if prob > prob_threshold:
                action_indexes.append(i)

        # If no actions meet threshold, return highest probability action
        if len(action_indexes) == 0:
            action_indexes = [int(np.argmax(probs_list))]

        return action_indexes, probs[0]

    def save_checkpoint(self, file_name : str):
        """Save model.

        Args:
            file_name (str): File name
        """
        torch.save(self.state_dict(), file_name)

    def load_checkpoint(self, file_name : str):
        """Load model

        Args:
            file_name (str): File name
        """
        self.load_state_dict(torch.load(file_name))

class ReinforceAgent(StochasticAgent):
    """
    Classical REINFORCE Agent with MLP Policy.

    Implements the REINFORCE algorithm using a classical neural network
    as the policy. This serves as a baseline for comparison with quantum
    implementations.

    Key features:
    - MLP policy with backpropagation
    - Monte Carlo policy gradient (episodic learning)
    - Stochastic policy for exploration and model checking
    - Return normalization for variance reduction
    - Entropy bonus for exploration
    """

    def __init__(self, state_dimension, hidden_layer_size, number_of_actions, gamma, lr, entropy_coef=0.01):
        """
        Initialize Classical REINFORCE Agent.

        Args:
            state_dimension: State space dimension
            hidden_layer_size: Hidden layer size for MLP
            number_of_actions: Action space dimension
            gamma: Discount factor
            lr: Learning rate
            entropy_coef: Entropy coefficient for exploration bonus
        """
        super().__init__()
        self.policy = Policy(state_dimension, hidden_layer_size, number_of_actions, lr)
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.rewards = []
        self.saved_log_probs = []
        self.saved_entropies = []

    def select_action(self, state: np.ndarray, deploy: bool = False):
        """
        Select action using current policy.

        Args:
            state: Current state
            deploy: If True, use greedy policy (argmax)

        Returns:
            Selected action index
        """
        action_index, log_prob, max_action = self.policy.act(state)

        if deploy:
            # Greedy action for deployment
            return max_action
        else:
            # Stochastic action for training
            # Store log probability for learning
            self.saved_log_probs.append(log_prob)

            # Compute entropy for exploration bonus
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(DEVICE)
            probs = self.policy.forward(state_tensor).cpu()
            dist = Categorical(probs)
            entropy = dist.entropy()
            self.saved_entropies.append(entropy)

            return action_index



    def model_checking_select_action(self, state: np.ndarray, prob_threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select actions stochastically for model checking (DTMC building).

        Args:
            state: Current state
            prob_threshold: Minimum probability threshold

        Returns:
            Tuple of (action_indices, probabilities)
        """
        return self.policy.stochastic_act(state, prob_threshold)

    def get_action_name_probability(self, env, action: str, state: np.ndarray) -> float:
        """
        Get probability of specific action for model checking.

        Args:
            env: Environment (for action name mapping)
            action: Name of action
            state: Current state

        Returns:
            Probability of action
        """
        with torch.no_grad():
            # Get action index from name
            action_idx = env.action_mapper.action_name_to_action_index(action)

            # Get action probabilities
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(DEVICE)
            probs = self.policy.forward(state_tensor).cpu()

            # Access probability directly with tensor indexing
            return float(probs[0, action_idx].item())







    def store_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminal: bool):
        """
        Store reward (REINFORCE only stores rewards, not full transitions).

        Args:
            state: Current state (not used)
            action: Action taken (not used)
            reward: Reward received
            next_state: Next state (not used)
            terminal: Terminal flag (not used)
        """
        self.rewards.append(reward)

    def episodic_learn(self):
        """
        Update policy using Monte Carlo returns and policy gradient.

        This implements the REINFORCE algorithm with improvements:
        1. Compute discounted returns G_t for each timestep
        2. Normalize returns (reduce variance)
        3. Compute policy gradient: ∇J(θ) = E[∇log π(a|s) * G_t]
        4. Update parameters using gradient ascent (maximize expected return)
        5. Add entropy bonus for exploration
        """
        if len(self.rewards) == 0:
            return

        # Compute discounted returns for each timestep (not just total return)
        # This is the correct REINFORCE implementation
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        # Convert to tensor
        returns = torch.tensor(returns, dtype=torch.float32)

        # Normalize returns (reduce variance) - important for stable learning
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy loss (negative because we're maximizing)
        # REINFORCE: -log π(a|s) * G_t for each timestep
        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)

        policy_loss = torch.stack(policy_loss).sum()

        # Add entropy bonus for exploration (negative because we're adding to loss)
        if len(self.saved_entropies) > 0:
            entropy_bonus = torch.stack(self.saved_entropies).sum()
            total_loss = policy_loss - self.entropy_coef * entropy_bonus
        else:
            total_loss = policy_loss

        # Gradient descent
        self.policy.optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

        self.policy.optimizer.step()

        # Clear episode memory
        self.rewards = []
        self.saved_log_probs = []
        self.saved_entropies = []



    def save(self, artifact_path='model'):
        """
        Save model to MLflow.

        Args:
            artifact_path: Artifact path for MLFlow (default: 'model')
        """
        try:
            os.mkdir('tmp_model')
        except Exception:
            pass

        # Save model state
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.policy.optimizer.state_dict(),
            'hyperparameters': {
                'gamma': self.gamma,
                'entropy_coef': self.entropy_coef
            }
        }, 'tmp_model/reinforce.pth')

        # Log to MLflow
        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)

        # Cleanup
        shutil.rmtree('tmp_model')

    def load(self, model_root_folder_path: str):
        """
        Load model from MLflow.

        Args:
            model_root_folder_path (str): Root folder containing the model checkpoint
        """
        if model_root_folder_path is None:
            return

        # Remove file:// prefix if present
        model_root_folder_path = model_root_folder_path.replace("file://", "")
        checkpoint_path = os.path.join(model_root_folder_path, 'reinforce.pth')

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded Classical REINFORCE model from {checkpoint_path}")
        else:
            print(f"Warning: Model checkpoint not found at {checkpoint_path}")
