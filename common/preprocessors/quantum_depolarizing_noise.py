"""
Quantum Depolarizing Noise Preprocessor

This preprocessor adds depolarizing noise to quantum agents during model checking.
Depolarizing noise is one of the most common quantum error models, representing
random Pauli errors (X, Y, Z) applied to qubits.

The noise is applied at the GATE LEVEL during quantum circuit execution,
meaning noise channels are inserted after each quantum gate operation.
This provides physically accurate simulation of noisy quantum hardware.

Configuration string format: "quantum_depolarizing_noise:p"
Example: "quantum_depolarizing_noise:0.01" for 1% depolarizing noise per gate
"""

import numpy as np
from common.preprocessors.preprocessor import Preprocessor


class QuantumDepolarizingNoise(Preprocessor):
    """
    Preprocessor that adds depolarizing noise to quantum agent circuits.

    This works by configuring the quantum agent to insert depolarizing channels
    after each gate operation in the quantum circuit. The noise accumulates
    through the circuit, providing physically accurate noise simulation.
    """

    def __init__(self, state_mapper, config_str, task):
        """
        Initialize the quantum depolarizing noise preprocessor.

        Args:
            state_mapper: State mapper for the environment
            config_str: Configuration string in format "quantum_depolarizing_noise:p"
            task: Task type (safe_training or rl_model_checking)
        """
        super().__init__(state_mapper, config_str, task)
        self.depolarizing_p = 0.0
        self.noise_configured = False
        self.parse_config(config_str)

    def parse_config(self, config_str: str) -> None:
        """
        Parse the configuration string to extract depolarizing parameter.

        Args:
            config_str: Format "quantum_depolarizing_noise:p" where p is in [0,1]
                       Example: "quantum_depolarizing_noise:0.01"
        """
        if ':' in config_str:
            parts = config_str.split(':')
            if len(parts) >= 2:
                try:
                    self.depolarizing_p = float(parts[1])
                    if not 0 <= self.depolarizing_p <= 1:
                        raise ValueError(f"Depolarizing parameter must be in [0,1], got {self.depolarizing_p}")
                    print(f"Quantum Depolarizing Noise: p={self.depolarizing_p}")
                except ValueError as e:
                    print(f"Error parsing depolarizing parameter: {e}")
                    self.depolarizing_p = 0.0
        else:
            print("Warning: No depolarizing parameter specified, using p=0.0 (no noise)")
            self.depolarizing_p = 0.0

    def _configure_agent_noise(self, agent):
        """
        Configure the quantum agent to use gate-level noise.

        This sets the agent's noise parameters, which will cause depolarizing
        channels to be inserted after each gate operation in the quantum circuit.

        Args:
            agent: The quantum RL agent to configure
        """
        if self.noise_configured:
            return

        # Check if agent has noise configuration attributes
        if hasattr(agent, 'noise_enabled') and hasattr(agent, 'depolarizing_p'):
            # Check if agent has enable_noise method (Quantum PPO V2)
            if hasattr(agent, 'enable_noise'):
                # Use the enable_noise method to properly reconfigure circuits
                agent.enable_noise(
                    depolarizing_p=self.depolarizing_p,
                    amplitude_damping_gamma=agent.amplitude_damping_gamma  # Keep existing amplitude damping
                )
            else:
                # Older agents (QuantumEvAgent) - just set attributes
                agent.noise_enabled = True
                agent.depolarizing_p = self.depolarizing_p

            self.noise_configured = True
            print(f"Configured quantum agent with gate-level depolarizing noise (p={self.depolarizing_p} per gate)")
            print(f"  Device: default.mixed (density matrix simulation)")
        else:
            print("Warning: Agent does not support gate-level noise configuration.")
            print("         This preprocessor requires a quantum agent with noise_enabled attribute.")

    def preprocess(self, rl_agent, state: np.ndarray, action_mapper, current_action_name: str, deploy: bool) -> np.ndarray:
        """
        Preprocess the state (configure agent noise on first call).

        This method is called during incremental model building. On the first call,
        it configures the quantum agent to use gate-level noise. Noise channels
        are then inserted after each gate operation during circuit execution.

        Args:
            rl_agent: The RL agent (should be QuantumEvAgent)
            state: The current state
            action_mapper: Action mapper from environment
            current_action_name: Current action name during incremental building
            deploy: Whether in deployment mode

        Returns:
            The state unchanged (noise is applied at circuit level)
        """
        # Configure agent noise on first call
        if not self.noise_configured:
            self._configure_agent_noise(rl_agent)

        # Return state unchanged (we modify the quantum circuit, not state)
        return state
