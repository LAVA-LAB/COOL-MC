"""
Generalized Amplitude Damping Noise Preprocessor for Quantum RL

This preprocessor adds generalized amplitude damping noise to quantum RL agents.
Generalized amplitude damping models energy relaxation (T1) at finite temperature,
including both spontaneous emission and thermal excitation.

Physical interpretation:
- gamma: Damping rate (related to T1 relaxation time)
- p: Thermal excitation probability (related to temperature)
- At p=0, reduces to standard amplitude damping (zero temperature)
- At p>0, models thermal environment that can also excite qubits

Kraus operators:
E_0 = sqrt(p) * [[1, 0], [0, sqrt(1-gamma)]]
E_1 = sqrt(p) * [[0, sqrt(gamma)], [0, 0]]
E_2 = sqrt(1-p) * [[sqrt(1-gamma), 0], [0, 1]]
E_3 = sqrt(1-p) * [[0, 0], [sqrt(gamma), 0]]

Usage:
    python cool_mc.py --preprocessor "quantum_generalized_amplitude_damping:0.01:0.1" ...
    # gamma=0.01 (damping rate), p=0.1 (thermal excitation probability)
"""

from common.preprocessors.preprocessor import Preprocessor


class QuantumGeneralizedAmplitudeDamping(Preprocessor):
    """
    Generalized Amplitude Damping noise preprocessor.

    This adds generalized amplitude damping noise to quantum REINFORCE agents,
    modeling T1 relaxation in a thermal environment.
    """

    def __init__(self, state_mapper, config_str, task):
        """
        Initialize the quantum generalized amplitude damping noise preprocessor.

        Args:
            state_mapper: State mapper for the environment
            config_str: Configuration string in format "quantum_generalized_amplitude_damping:gamma:p"
            task: Task type (safe_training or rl_model_checking)
        """
        super().__init__(state_mapper, config_str, task)
        self.gamma = 0.0
        self.p = 0.0
        self.noise_configured = False
        self.parse_config(config_str)

    def parse_config(self, config_str: str) -> None:
        """
        Parse the configuration string to extract generalized amplitude damping parameters.

        Args:
            config_str: Format "quantum_generalized_amplitude_damping:gamma:p"
                       where gamma is damping rate [0,1] and p is thermal probability [0,1]
                       Example: "quantum_generalized_amplitude_damping:0.01:0.1"
        """
        if ':' in config_str:
            parts = config_str.split(':')
            if len(parts) >= 2:
                try:
                    self.gamma = float(parts[1])
                    if not 0 <= self.gamma <= 1:
                        raise ValueError(f"Gamma must be in [0,1], got {self.gamma}")
                    if len(parts) >= 3:
                        self.p = float(parts[2])
                        if not 0 <= self.p <= 1:
                            raise ValueError(f"p must be in [0,1], got {self.p}")
                    print(f"Quantum Generalized Amplitude Damping: gamma={self.gamma}, p={self.p}")
                except ValueError as e:
                    print(f"Error parsing generalized amplitude damping parameters: {e}")
                    self.gamma = 0.0
                    self.p = 0.0
        else:
            print("Warning: No parameters specified, using gamma=0.0, p=0.0 (no noise)")
            self.gamma = 0.0
            self.p = 0.0

    def _configure_agent_noise(self, agent):
        """Configure agent with generalized amplitude damping noise."""
        if hasattr(agent, 'enable_noise'):
            # Check if agent already has noise enabled and preserve existing noise
            depolarizing_p = getattr(agent, 'depolarizing_p', 0.0)
            amplitude_damping_gamma = getattr(agent, 'amplitude_damping_gamma', 0.0)
            phase_damping_gamma = getattr(agent, 'phase_damping_gamma', 0.0)
            bit_flip_p = getattr(agent, 'bit_flip_p', 0.0)
            phase_flip_p = getattr(agent, 'phase_flip_p', 0.0)

            agent.enable_noise(
                depolarizing_p=depolarizing_p,
                amplitude_damping_gamma=amplitude_damping_gamma,
                phase_damping_gamma=phase_damping_gamma,
                bit_flip_p=bit_flip_p,
                phase_flip_p=phase_flip_p,
                generalized_amplitude_damping_gamma=self.gamma,
                generalized_amplitude_damping_p=self.p
            )

    def __call__(self, agent, state):
        """
        Apply generalized amplitude damping to agent's quantum circuit.

        Args:
            agent: Quantum RL agent (must have enable_noise method)
            state: Environment state (returned unchanged)

        Returns:
            state: Unmodified environment state
        """
        self._configure_agent_noise(agent)
        return state
