"""
Quantum Phase Flip Noise Preprocessor

This preprocessor adds phase flip noise to quantum agents during model checking.
Phase flip noise applies a Pauli-Z error with probability p, which flips the
phase of the |1⟩ state without changing populations.

The noise is applied at the GATE LEVEL during quantum circuit execution,
meaning phase flip channels are inserted after each quantum gate operation.

Configuration string format: "quantum_phase_flip:p"
Example: "quantum_phase_flip:0.01" for 1% phase flip probability per gate

Mathematical definition:
E_PF(ρ) = (1 − p)ρ + p ZρZ
where Z is the Pauli-Z operator and p is the phase-flip probability.
"""

import numpy as np
from common.preprocessors.preprocessor import Preprocessor


class QuantumPhaseFlip(Preprocessor):
    """
    Preprocessor that adds phase flip noise to quantum agent circuits.

    This works by configuring the quantum agent to insert phase flip channels
    after each gate operation in the quantum circuit. Phase flip noise applies
    the Pauli-Z operator (phase flip: |+⟩ ↔ |−⟩) with probability p.
    """

    def __init__(self, state_mapper, config_str, task):
        """
        Initialize the quantum phase flip noise preprocessor.

        Args:
            state_mapper: State mapper for the environment
            config_str: Configuration string in format "quantum_phase_flip:p"
            task: Task type (safe_training or rl_model_checking)
        """
        super().__init__(state_mapper, config_str, task)
        self.p = 0.0
        self.noise_configured = False
        self.parse_config(config_str)

    def parse_config(self, config_str: str) -> None:
        """
        Parse the configuration string to extract phase flip parameter.

        Args:
            config_str: Format "quantum_phase_flip:p" where p is in [0,1]
                       Example: "quantum_phase_flip:0.01"
        """
        if ':' in config_str:
            parts = config_str.split(':')
            if len(parts) >= 2:
                try:
                    self.p = float(parts[1])
                    if not 0 <= self.p <= 1:
                        raise ValueError(f"Phase flip parameter must be in [0,1], got {self.p}")
                    print(f"Quantum Phase Flip: p={self.p}")
                except ValueError as e:
                    print(f"Error parsing phase flip parameter: {e}")
                    self.p = 0.0
        else:
            print("Warning: No phase flip parameter specified, using p=0.0 (no noise)")
            self.p = 0.0

    def _configure_agent_noise(self, agent):
        """
        Configure the quantum agent to use gate-level phase flip noise.

        This sets the agent's noise parameters, which will cause phase flip
        channels to be inserted after each gate operation in the quantum circuit.

        Args:
            agent: The quantum RL agent to configure
        """
        if self.noise_configured:
            return

        # Check if agent has noise configuration attributes
        if hasattr(agent, 'noise_enabled') and hasattr(agent, 'phase_flip_p'):
            # Check if agent has enable_noise method
            if hasattr(agent, 'enable_noise'):
                # Get existing noise parameters
                depolarizing_p = getattr(agent, 'depolarizing_p', 0.0)
                amplitude_damping_gamma = getattr(agent, 'amplitude_damping_gamma', 0.0)
                phase_damping_gamma = getattr(agent, 'phase_damping_gamma', 0.0)
                bit_flip_p = getattr(agent, 'bit_flip_p', 0.0)

                # Use the enable_noise method to properly reconfigure circuits
                agent.enable_noise(
                    depolarizing_p=depolarizing_p,
                    amplitude_damping_gamma=amplitude_damping_gamma,
                    phase_damping_gamma=phase_damping_gamma,
                    bit_flip_p=bit_flip_p,
                    phase_flip_p=self.p
                )
            else:
                # Older agents - just set attributes
                agent.noise_enabled = True
                agent.phase_flip_p = self.p

            self.noise_configured = True
            print(f"Configured quantum agent with gate-level phase flip noise (p={self.p} per gate)")
            print(f"  Device: default.mixed (density matrix simulation)")
            print(f"  Effect: Applies Pauli-Z with probability p (phase flip: |+⟩ ↔ |−⟩)")
        else:
            print("Warning: Agent does not support phase flip noise configuration.")
            print("         This preprocessor requires a quantum agent with phase_flip_p attribute.")

    def preprocess(self, rl_agent, state: np.ndarray, action_mapper, current_action_name: str, deploy: bool) -> np.ndarray:
        """
        Preprocess the state (configure agent noise on first call).

        This method is called during incremental model building. On the first call,
        it configures the quantum agent to use gate-level phase flip noise.
        Phase flip channels are then inserted after each gate operation during circuit execution.

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
