"""
Quantum Bit Flip Noise Preprocessor

This preprocessor adds bit flip noise to quantum agents during model checking.
Bit flip noise applies a Pauli-X error with probability p, modeling the classical
bit flip error in a quantum setting.

The noise is applied at the GATE LEVEL during quantum circuit execution,
meaning bit flip channels are inserted after each quantum gate operation.

Configuration string format: "quantum_bit_flip:p"
Example: "quantum_bit_flip:0.01" for 1% bit flip probability per gate

Mathematical definition:
E_BF(ρ) = (1 − p)ρ + p XρX
where X is the Pauli-X operator and p is the bit-flip probability.
"""

import numpy as np
from common.preprocessors.preprocessor import Preprocessor


class QuantumBitFlip(Preprocessor):
    """
    Preprocessor that adds bit flip noise to quantum agent circuits.

    This works by configuring the quantum agent to insert bit flip channels
    after each gate operation in the quantum circuit. Bit flip noise applies
    the Pauli-X operator (|0⟩ ↔ |1⟩ swap) with probability p.
    """

    def __init__(self, state_mapper, config_str, task):
        """
        Initialize the quantum bit flip noise preprocessor.

        Args:
            state_mapper: State mapper for the environment
            config_str: Configuration string in format "quantum_bit_flip:p"
            task: Task type (safe_training or rl_model_checking)
        """
        super().__init__(state_mapper, config_str, task)
        self.p = 0.0
        self.noise_configured = False
        self.parse_config(config_str)

    def parse_config(self, config_str: str) -> None:
        """
        Parse the configuration string to extract bit flip parameter.

        Args:
            config_str: Format "quantum_bit_flip:p" where p is in [0,1]
                       Example: "quantum_bit_flip:0.01"
        """
        if ':' in config_str:
            parts = config_str.split(':')
            if len(parts) >= 2:
                try:
                    self.p = float(parts[1])
                    if not 0 <= self.p <= 1:
                        raise ValueError(f"Bit flip parameter must be in [0,1], got {self.p}")
                    print(f"Quantum Bit Flip: p={self.p}")
                except ValueError as e:
                    print(f"Error parsing bit flip parameter: {e}")
                    self.p = 0.0
        else:
            print("Warning: No bit flip parameter specified, using p=0.0 (no noise)")
            self.p = 0.0

    def _configure_agent_noise(self, agent):
        """
        Configure the quantum agent to use gate-level bit flip noise.

        This sets the agent's noise parameters, which will cause bit flip
        channels to be inserted after each gate operation in the quantum circuit.

        Args:
            agent: The quantum RL agent to configure
        """
        if self.noise_configured:
            return

        # Check if agent has noise configuration attributes
        if hasattr(agent, 'noise_enabled') and hasattr(agent, 'bit_flip_p'):
            # Check if agent has enable_noise method
            if hasattr(agent, 'enable_noise'):
                # Get existing noise parameters
                depolarizing_p = getattr(agent, 'depolarizing_p', 0.0)
                amplitude_damping_gamma = getattr(agent, 'amplitude_damping_gamma', 0.0)
                phase_damping_gamma = getattr(agent, 'phase_damping_gamma', 0.0)
                phase_flip_p = getattr(agent, 'phase_flip_p', 0.0)

                # Use the enable_noise method to properly reconfigure circuits
                agent.enable_noise(
                    depolarizing_p=depolarizing_p,
                    amplitude_damping_gamma=amplitude_damping_gamma,
                    phase_damping_gamma=phase_damping_gamma,
                    bit_flip_p=self.p,
                    phase_flip_p=phase_flip_p
                )
            else:
                # Older agents - just set attributes
                agent.noise_enabled = True
                agent.bit_flip_p = self.p

            self.noise_configured = True
            print(f"Configured quantum agent with gate-level bit flip noise (p={self.p} per gate)")
            print(f"  Device: default.mixed (density matrix simulation)")
            print(f"  Effect: Applies Pauli-X with probability p (|0⟩ ↔ |1⟩ swap)")
        else:
            print("Warning: Agent does not support bit flip noise configuration.")
            print("         This preprocessor requires a quantum agent with bit_flip_p attribute.")

    def preprocess(self, rl_agent, state: np.ndarray, action_mapper, current_action_name: str, deploy: bool) -> np.ndarray:
        """
        Preprocess the state (configure agent noise on first call).

        This method is called during incremental model building. On the first call,
        it configures the quantum agent to use gate-level bit flip noise.
        Bit flip channels are then inserted after each gate operation during circuit execution.

        Args:
            rl_agent: The RL agent (should be a quantum agent)
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
