"""
Quantum Amplitude Damping Noise Preprocessor

This preprocessor adds amplitude damping noise to quantum agents during model checking.
Amplitude damping represents energy loss (T1 relaxation), one of the most important
quantum noise mechanisms in real quantum hardware.

The noise is applied at the GATE LEVEL during quantum circuit execution,
meaning amplitude damping channels are inserted after each quantum gate operation.
This provides physically accurate simulation of energy relaxation in noisy quantum hardware.

Configuration string format: "quantum_amplitude_damping:gamma"
Example: "quantum_amplitude_damping:0.01" for 1% energy loss per gate
"""

import numpy as np
from common.preprocessors.preprocessor import Preprocessor


class QuantumAmplitudeDamping(Preprocessor):
    """
    Preprocessor that adds amplitude damping noise to quantum agent circuits.

    This works by configuring the quantum agent to insert amplitude damping channels
    after each gate operation in the quantum circuit. The noise models energy loss
    (qubits decaying from |1⟩ to |0⟩), providing physically accurate T1 relaxation simulation.
    """

    def __init__(self, state_mapper, config_str, task):
        """
        Initialize the quantum amplitude damping noise preprocessor.

        Args:
            state_mapper: State mapper for the environment
            config_str: Configuration string in format "quantum_amplitude_damping:gamma"
            task: Task type (safe_training or rl_model_checking)
        """
        super().__init__(state_mapper, config_str, task)
        self.gamma = 0.0
        self.noise_configured = False
        self.parse_config(config_str)

    def parse_config(self, config_str: str) -> None:
        """
        Parse the configuration string to extract amplitude damping parameter.

        Args:
            config_str: Format "quantum_amplitude_damping:gamma" where gamma is in [0,1]
                       Example: "quantum_amplitude_damping:0.01"
        """
        if ':' in config_str:
            parts = config_str.split(':')
            if len(parts) >= 2:
                try:
                    self.gamma = float(parts[1])
                    if not 0 <= self.gamma <= 1:
                        raise ValueError(f"Amplitude damping parameter must be in [0,1], got {self.gamma}")
                    print(f"Quantum Amplitude Damping: gamma={self.gamma}")
                except ValueError as e:
                    print(f"Error parsing amplitude damping parameter: {e}")
                    self.gamma = 0.0
        else:
            print("Warning: No amplitude damping parameter specified, using gamma=0.0 (no noise)")
            self.gamma = 0.0

    def _configure_agent_noise(self, agent):
        """
        Configure the quantum agent to use gate-level amplitude damping noise.

        This sets the agent's noise parameters, which will cause amplitude damping
        channels to be inserted after each gate operation in the quantum circuit.

        Args:
            agent: The quantum RL agent to configure
        """
        if self.noise_configured:
            return

        # Check if agent has noise configuration attributes
        if hasattr(agent, 'noise_enabled') and hasattr(agent, 'amplitude_damping_gamma'):
            # Check if agent has enable_noise method (Quantum PPO V2)
            if hasattr(agent, 'enable_noise'):
                # Use the enable_noise method to properly reconfigure circuits
                agent.enable_noise(
                    depolarizing_p=agent.depolarizing_p,  # Keep existing depolarizing
                    amplitude_damping_gamma=self.gamma
                )
            else:
                # Older agents (QuantumEvAgent) - just set attributes
                agent.noise_enabled = True
                agent.amplitude_damping_gamma = self.gamma

            self.noise_configured = True
            print(f"Configured quantum agent with gate-level amplitude damping noise (gamma={self.gamma} per gate)")
            print(f"  Device: default.mixed (density matrix simulation)")
            print(f"  Effect: Models energy relaxation (T1 decay) - qubits decay from |1⟩ to |0⟩")
        else:
            print("Warning: Agent does not support amplitude damping noise configuration.")
            print("         This preprocessor requires a quantum agent with amplitude_damping_gamma attribute.")

    def preprocess(self, rl_agent, state: np.ndarray, action_mapper, current_action_name: str, deploy: bool) -> np.ndarray:
        """
        Preprocess the state (configure agent noise on first call).

        This method is called during incremental model building. On the first call,
        it configures the quantum agent to use gate-level amplitude damping noise.
        Amplitude damping channels are then inserted after each gate operation during circuit execution.

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
