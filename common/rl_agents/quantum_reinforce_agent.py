"""
Quantum REINFORCE Agent with Variational Quantum Circuit and Parameter Shift Rule

This agent implements a quantum version of the REINFORCE (Monte Carlo Policy Gradient) algorithm:
- Variational Quantum Circuit (VQC) as the policy network
- Parameter Shift Rule for gradient calculation (quantum-native gradients)
- Amplitude encoding for state representation
- Hardware-efficient ansatz with rotation gates and entanglement
- Episodic learning with Monte Carlo returns

References:
- REINFORCE: Simple Statistical Gradient-Following Algorithms (Williams, 1992)
- Variational Quantum Algorithms (McClean et al., 2016)
- Parameter Shift Rule (Mitarai et al., 2018; Schuld et al., 2019)
- Quantum Circuit Learning (Farhi & Neven, 2018)
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import mlflow
import os
import shutil
from common.rl_agents.agent import StochasticAgent
from typing import Tuple, List


class QuantumPolicyNetwork(nn.Module):
    """
    Quantum Policy Network using Variational Quantum Circuit.

    Architecture:
    1. Amplitude encoding: Classical state -> Quantum state
    2. Variational layers: Parameterized rotations + entanglement
    3. Measurement: Pauli-Z expectation values
    4. Classical post-processing: Map measurements to action probabilities

    The parameter shift rule is used for gradient calculation, which is
    a quantum-native method that computes exact gradients by evaluating
    the circuit at shifted parameter values.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_qubits: int = 4,
        num_layers: int = 3,
        use_parameter_shift: bool = True
    ):
        """
        Initialize Quantum Policy Network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            num_qubits: Number of qubits in quantum circuit
            num_layers: Number of variational layers
            use_parameter_shift: If True, use parameter shift rule for gradients
                                If False, use backpropagation
        """
        super(QuantumPolicyNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.use_parameter_shift = use_parameter_shift
        self.state_size = 2 ** num_qubits  # Quantum state size

        # Number of parameters per layer: 3 rotations per qubit (RX, RY, RZ)
        self.params_per_layer = num_qubits * 3
        self.total_params = self.params_per_layer * num_layers

        # Quantum noise parameters (for model checking with noise)
        self.noise_enabled = False
        self.depolarizing_p = 0.0
        self.amplitude_damping_gamma = 0.0
        self.phase_damping_gamma = 0.0
        self.bit_flip_p = 0.0
        self.phase_flip_p = 0.0
        self.generalized_amplitude_damping_gamma = 0.0
        self.generalized_amplitude_damping_p = 0.0  # Thermal excitation probability

        # Initialize quantum device
        self._init_device()

        # Trainable quantum parameters (variational parameters)
        # Initialize with small random values for better training
        self.circuit_params = nn.Parameter(
            torch.randn(self.total_params) * 0.01
        )

        # Classical post-processing layer: map quantum measurements to action probabilities
        # This is a simple linear layer that combines qubit measurements
        self.classical_head = nn.Linear(num_qubits, action_dim)

        # Create quantum circuit with appropriate differentiation method
        self._create_quantum_circuit()

    def _init_device(self):
        """Initialize quantum device (default.qubit or default.mixed for noise)."""
        if self.noise_enabled:
            # Use default.mixed for density matrix simulation (supports noise)
            self.dev = qml.device('default.mixed', wires=self.num_qubits)
        else:
            # Use default.qubit for clean state vector simulation
            self.dev = qml.device('default.qubit', wires=self.num_qubits)

    def enable_noise(self, depolarizing_p=0.0, amplitude_damping_gamma=0.0, phase_damping_gamma=0.0,
                     bit_flip_p=0.0, phase_flip_p=0.0, generalized_amplitude_damping_gamma=0.0,
                     generalized_amplitude_damping_p=0.0):
        """
        Enable quantum noise and recreate circuit.

        Args:
            depolarizing_p: Depolarizing noise parameter (0.0 to 1.0)
            amplitude_damping_gamma: Amplitude damping parameter (0.0 to 1.0)
            phase_damping_gamma: Phase damping parameter (0.0 to 1.0)
            bit_flip_p: Bit flip probability (0.0 to 1.0)
            phase_flip_p: Phase flip probability (0.0 to 1.0)
            generalized_amplitude_damping_gamma: Generalized amplitude damping gamma (0.0 to 1.0)
            generalized_amplitude_damping_p: Thermal excitation probability (0.0 to 1.0)
        """
        self.noise_enabled = True
        self.depolarizing_p = depolarizing_p
        self.amplitude_damping_gamma = amplitude_damping_gamma
        self.phase_damping_gamma = phase_damping_gamma
        self.bit_flip_p = bit_flip_p
        self.phase_flip_p = phase_flip_p
        self.generalized_amplitude_damping_gamma = generalized_amplitude_damping_gamma
        self.generalized_amplitude_damping_p = generalized_amplitude_damping_p

        # Recreate device and circuit with noise support
        self._init_device()
        self._create_quantum_circuit()

    def _inject_noise(self, qubit):
        """
        Inject all enabled noise channels on a single qubit.

        This helper method consolidates noise injection to avoid code duplication.
        Noise channels are applied in order: depolarizing, bit/phase flips, damping,
        generalized damping. All noise types are time-independent and probability-based.

        Args:
            qubit: Qubit index to apply noise to
        """
        # Standard Pauli noise channels
        if self.depolarizing_p > 0:
            qml.DepolarizingChannel(self.depolarizing_p, wires=qubit)
        if self.bit_flip_p > 0:
            qml.BitFlip(self.bit_flip_p, wires=qubit)
        if self.phase_flip_p > 0:
            qml.PhaseFlip(self.phase_flip_p, wires=qubit)

        # Damping channels (time-independent, probability-based)
        if self.amplitude_damping_gamma > 0:
            qml.AmplitudeDamping(self.amplitude_damping_gamma, wires=qubit)
        if self.phase_damping_gamma > 0:
            qml.PhaseDamping(self.phase_damping_gamma, wires=qubit)

        # Generalized amplitude damping (includes thermal excitation, time-independent)
        if self.generalized_amplitude_damping_gamma > 0:
            qml.GeneralizedAmplitudeDamping(
                self.generalized_amplitude_damping_gamma,
                self.generalized_amplitude_damping_p,
                wires=qubit
            )

    def _amplitude_encode_state(self, state: np.ndarray) -> np.ndarray:
        """
        Encode classical state into quantum amplitudes.

        Uses amplitude encoding: state vector becomes quantum amplitudes.
        The state is normalized to form a valid quantum state.

        Args:
            state: Classical state vector

        Returns:
            Normalized quantum state (amplitudes)
        """
        # Create quantum state vector
        quantum_state = np.zeros(self.state_size)

        if len(state) <= self.state_size:
            # Simple padding
            quantum_state[:len(state)] = state
        else:
            # Feature map for high-dimensional states
            for i, val in enumerate(state):
                quantum_state[i % self.state_size] += val

        # Normalize to unit vector (required for quantum states)
        norm = np.linalg.norm(quantum_state)
        if norm > 0:
            quantum_state = quantum_state / norm
        else:
            # Default to |0...0⟩ state
            quantum_state[0] = 1.0

        return quantum_state

    def _create_quantum_circuit(self):
        """
        Create parameterized quantum circuit for policy.

        Uses parameter shift rule or backpropagation for gradient calculation.
        Parameter shift rule is the quantum-native approach that evaluates
        the circuit at shifted parameter values to compute exact gradients.
        """
        # Select differentiation method
        if self.use_parameter_shift:
            # Parameter shift rule: quantum-native gradient calculation
            # Computes exact gradients by evaluating circuit at shifted parameters
            diff_method = 'parameter-shift'
        else:
            # Backpropagation: uses PyTorch autograd
            diff_method = 'backprop'

        @qml.qnode(self.dev, interface='torch', diff_method=diff_method)
        def policy_circuit(state_amplitudes, params):
            """
            Parameterized Quantum Circuit for Policy Network.

            Architecture:
            1. Amplitude encoding: Initialize quantum state with input
            2. Variational layers: Rotation gates (RX, RY, RZ) + entanglement (CNOT)
            3. Measurement: PauliZ expectation values

            The parameter shift rule works by computing derivatives as:
            ∂⟨ψ|O|ψ⟩/∂θ = (⟨ψ(θ+π/2)|O|ψ(θ+π/2)⟩ - ⟨ψ(θ-π/2)|O|ψ(θ-π/2)⟩) / 2

            Args:
                state_amplitudes: Quantum state amplitudes (normalized)
                params: Variational parameters (rotation angles)

            Returns:
                Z expectation values for each qubit
            """
            # 1. Amplitude encoding - prepare quantum state
            qml.StatePrep(state_amplitudes, wires=range(self.num_qubits))

            # 2. Variational layers (hardware-efficient ansatz)
            param_idx = 0
            for layer in range(self.num_layers):
                # Rotation layer - individual qubit rotations
                for qubit in range(self.num_qubits):
                    qml.RX(params[param_idx], wires=qubit)
                    # Add noise after gate if enabled
                    if self.noise_enabled:
                        self._inject_noise(qubit)

                    qml.RY(params[param_idx + 1], wires=qubit)
                    if self.noise_enabled:
                        self._inject_noise(qubit)

                    qml.RZ(params[param_idx + 2], wires=qubit)
                    if self.noise_enabled:
                        self._inject_noise(qubit)

                    param_idx += 3

                # Entangling layer - create quantum correlations
                if layer < self.num_layers - 1:  # Skip entanglement on last layer
                    for qubit in range(self.num_qubits - 1):
                        qml.CNOT(wires=[qubit, qubit + 1])
                        # Add noise after CNOT if enabled
                        if self.noise_enabled:
                            self._inject_noise(qubit)
                            self._inject_noise(qubit + 1)

                    # Ring topology: connect last to first
                    qml.CNOT(wires=[self.num_qubits - 1, 0])
                    if self.noise_enabled:
                        self._inject_noise(self.num_qubits - 1)
                        self._inject_noise(0)

            # 3. Measurements - Z expectation for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.policy_circuit = policy_circuit

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum policy network.

        Args:
            state: Input state tensor

        Returns:
            Action probabilities (softmax over actions)
        """
        # Convert to numpy for quantum encoding
        state_np = state.detach().cpu().numpy()

        # Amplitude encode state
        quantum_state = self._amplitude_encode_state(state_np)
        quantum_state_tensor = torch.tensor(quantum_state, dtype=torch.float32)

        # Run quantum circuit (gradients computed via parameter shift or backprop)
        measurements = self.policy_circuit(quantum_state_tensor, self.circuit_params)
        measurements = torch.stack(measurements).float()

        # Classical post-processing: measurements -> action logits
        action_logits = self.classical_head(measurements)

        # Convert to probabilities using softmax
        action_probs = torch.softmax(action_logits, dim=-1)

        return action_probs

    def act(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, int]:
        """
        Select action using current policy.

        Args:
            state: Input state

        Returns:
            Tuple of (sampled_action, log_prob, greedy_action)
        """
        action_probs = self.forward(state)

        # Sample action from policy distribution
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        # Also return greedy action (argmax)
        greedy_action = torch.argmax(action_probs)

        return action.item(), action_logprob, greedy_action.item()


class QuantumReinforceAgent(StochasticAgent):
    """
    Quantum REINFORCE Agent with Parameter Shift Rule.

    Implements the REINFORCE algorithm using a Variational Quantum Circuit (VQC)
    as the policy network. Gradients are computed using the parameter shift rule,
    which is a quantum-native method for computing exact gradients.

    Key features:
    - VQC policy with amplitude encoding
    - Parameter shift rule for gradient calculation
    - Monte Carlo policy gradient (episodic learning)
    - Stochastic policy for exploration and model checking
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_qubits: int = 4,
        num_layers: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        use_parameter_shift: bool = True,
        entropy_coef: float = 0.01
    ):
        """
        Initialize Quantum REINFORCE Agent.

        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            num_qubits: Number of qubits in quantum circuit
            num_layers: Number of variational layers
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            use_parameter_shift: If True, use parameter shift rule for gradients
            entropy_coef: Entropy coefficient for exploration bonus
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.use_parameter_shift = use_parameter_shift
        self.entropy_coef = entropy_coef

        # Quantum policy network
        self.policy = QuantumPolicyNetwork(
            state_dim, action_dim, num_qubits, num_layers, use_parameter_shift
        )

        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # Episode memory (for Monte Carlo returns)
        self.saved_log_probs = []
        self.saved_entropies = []
        self.rewards = []

        # Quantum noise parameters (accessible for preprocessors)
        self.noise_enabled = False
        self.depolarizing_p = 0.0
        self.amplitude_damping_gamma = 0.0
        self.phase_damping_gamma = 0.0
        self.bit_flip_p = 0.0
        self.phase_flip_p = 0.0

    def enable_noise(self, depolarizing_p=0.0, amplitude_damping_gamma=0.0, phase_damping_gamma=0.0, bit_flip_p=0.0, phase_flip_p=0.0):
        """
        Enable quantum noise in the policy network.

        This method is called by noise preprocessors to add realistic
        quantum noise for model checking.

        Args:
            depolarizing_p: Depolarizing noise parameter (0.0 to 1.0)
            amplitude_damping_gamma: Amplitude damping parameter (0.0 to 1.0)
            phase_damping_gamma: Phase damping parameter (0.0 to 1.0)
            bit_flip_p: Bit flip probability (0.0 to 1.0)
            phase_flip_p: Phase flip probability (0.0 to 1.0)
        """
        self.noise_enabled = True
        self.depolarizing_p = depolarizing_p
        self.amplitude_damping_gamma = amplitude_damping_gamma
        self.phase_damping_gamma = phase_damping_gamma
        self.bit_flip_p = bit_flip_p
        self.phase_flip_p = phase_flip_p

        # Enable noise in the policy network
        self.policy.enable_noise(depolarizing_p, amplitude_damping_gamma, phase_damping_gamma, bit_flip_p, phase_flip_p)

    def select_action(self, state: np.ndarray, deploy: bool = False) -> int:
        """
        Select action using current policy.

        Args:
            state: Current state
            deploy: If True, use greedy policy (argmax)

        Returns:
            Selected action index
        """
        state_tensor = torch.FloatTensor(state)

        if deploy:
            # Greedy action for deployment
            with torch.no_grad():
                _, _, greedy_action = self.policy.act(state_tensor)
                return greedy_action
        else:
            # Stochastic action for training
            action, log_prob, _ = self.policy.act(state_tensor)

            # Compute entropy for exploration bonus
            action_probs = self.policy.forward(state_tensor)
            dist = Categorical(action_probs)
            entropy = dist.entropy()

            # Store log probability and entropy for learning
            self.saved_log_probs.append(log_prob)
            self.saved_entropies.append(entropy)

            return action

    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminal: bool
    ):
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

        When using parameter shift rule, gradients are computed as:
        ∂L/∂θ_i = (L(θ + π/2*e_i) - L(θ - π/2*e_i)) / 2
        """
        if len(self.rewards) == 0:
            return

        # Compute discounted returns for each timestep (not just total return)
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

        # Gradient descent (parameter shift rule or backprop)
        self.optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Clear episode memory
        self.rewards = []
        self.saved_log_probs = []
        self.saved_entropies = []

    def model_checking_select_action(
        self,
        state: np.ndarray,
        prob_threshold: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select actions stochastically for model checking (DTMC building).

        Args:
            state: Current state
            prob_threshold: Minimum probability threshold

        Returns:
            Tuple of (action_indices, probabilities)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action_probs = self.policy.forward(state_tensor)
            probs_np = action_probs.numpy()

            # Filter actions by threshold
            valid_actions = np.where(probs_np >= prob_threshold)[0]

            if len(valid_actions) == 0:
                # If no actions meet threshold, return highest probability action
                valid_actions = np.array([np.argmax(probs_np)])

            return valid_actions, probs_np[valid_actions]

    def get_action_name_probability(
        self,
        env,
        action_name: str,
        state: np.ndarray
    ) -> float:
        """
        Get probability of specific action for model checking.

        Args:
            env: Environment (for action name mapping)
            action_name: Name of action
            state: Current state

        Returns:
            Probability of action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            action_probs = self.policy.forward(state_tensor)

            # Get action index from name
            action_idx = env.action_mapper.action_name_to_action_index(action_name)

            return float(action_probs[action_idx].item())

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
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'num_qubits': self.policy.num_qubits,
                'num_layers': self.policy.num_layers,
                'gamma': self.gamma,
                'use_parameter_shift': self.use_parameter_shift
            }
        }, 'tmp_model/quantum_reinforce.pth')

        # Log to MLflow
        mlflow.log_artifacts("tmp_model", artifact_path=artifact_path)

        # Cleanup
        shutil.rmtree('tmp_model')

    def load(self, root_folder: str):
        """
        Load model from MLflow.

        Args:
            root_folder: Root folder containing the model checkpoint
        """
        if root_folder is None:
            return

        # Remove file:// prefix if present
        root_folder = root_folder.replace("file://", "")
        checkpoint_path = os.path.join(root_folder, 'quantum_reinforce.pth')

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Loaded Quantum REINFORCE model from {checkpoint_path}")
        else:
            print(f"Warning: Model checkpoint not found at {checkpoint_path}")
