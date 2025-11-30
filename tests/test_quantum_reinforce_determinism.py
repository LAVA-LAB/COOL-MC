"""
Unit tests for Quantum REINFORCE Agent Determinism

This test suite verifies that the quantum REINFORCE agent produces deterministic,
reproducible probability distributions for model checking. This is critical because:
1. Model checking requires exact probability values to build the DTMC
2. Non-deterministic policies would lead to incorrect verification results
3. The same state must always produce the same action probabilities
"""

import unittest
import sys
import os
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.rl_agents.quantum_reinforce_agent import QuantumReinforceAgent, QuantumPolicyNetwork


class TestQuantumReinforceDeterminism(unittest.TestCase):
    """Test deterministic behavior of quantum REINFORCE agent."""

    def setUp(self):
        """Set up test fixtures."""
        # Set random seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)

        # Test parameters
        self.state_dim = 4
        self.action_dim = 2
        self.num_qubits = 4
        self.num_layers = 3

        # Create agent
        self.agent = QuantumReinforceAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_qubits=self.num_qubits,
            num_layers=self.num_layers,
            learning_rate=0.001,
            gamma=0.99,
            use_parameter_shift=True,
            entropy_coef=0.01
        )

        # Test states
        self.test_states = [
            np.array([0.5, 0.3, 0.2, 0.1]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([0.25, 0.25, 0.25, 0.25]),
            np.array([0.9, 0.05, 0.03, 0.02]),
            np.zeros(4)
        ]

    def test_policy_network_deterministic_forward_pass(self):
        """Test that policy network forward pass is deterministic."""
        state = torch.FloatTensor(self.test_states[0])

        # Get probability distributions multiple times
        probabilities = []
        for _ in range(10):
            with torch.no_grad():
                probs = self.agent.policy.forward(state)
                probabilities.append(probs.numpy().copy())

        # Verify all probability distributions are identical
        for i in range(1, len(probabilities)):
            np.testing.assert_array_almost_equal(
                probabilities[0],
                probabilities[i],
                decimal=10,
                err_msg=f"Forward pass {i} produced different probabilities than forward pass 0"
            )

    def test_model_checking_select_action_deterministic(self):
        """Test that model_checking_select_action returns deterministic probabilities."""
        for state_idx, state in enumerate(self.test_states):
            probabilities = []

            # Query the policy 100 times for the same state
            for _ in range(100):
                actions, probs = self.agent.model_checking_select_action(state, prob_threshold=0.0)
                probabilities.append(probs.copy())

            # Verify all probability distributions are identical
            for i in range(1, len(probabilities)):
                np.testing.assert_array_almost_equal(
                    probabilities[0],
                    probabilities[i],
                    decimal=10,
                    err_msg=f"State {state_idx}: Query {i} produced different probabilities"
                )

            print(f"✓ State {state_idx}: All 100 queries produced identical probabilities: {probabilities[0]}")

    def test_get_action_name_probability_deterministic(self):
        """Test that get_action_name_probability returns deterministic values."""
        # Mock environment with action mapper
        class MockActionMapper:
            def action_name_to_action_index(self, action_name):
                return int(action_name.split('_')[1])

        class MockEnv:
            def __init__(self):
                self.action_mapper = MockActionMapper()

        mock_env = MockEnv()

        for state_idx, state in enumerate(self.test_states):
            # Query probability for action 0 multiple times
            action_0_probs = []
            action_1_probs = []

            for _ in range(50):
                prob_0 = self.agent.get_action_name_probability(mock_env, "action_0", state)
                prob_1 = self.agent.get_action_name_probability(mock_env, "action_1", state)
                action_0_probs.append(prob_0)
                action_1_probs.append(prob_1)

            # Verify all probabilities are identical
            self.assertTrue(
                all(abs(p - action_0_probs[0]) < 1e-10 for p in action_0_probs),
                f"State {state_idx}: Action 0 probabilities varied across queries"
            )
            self.assertTrue(
                all(abs(p - action_1_probs[0]) < 1e-10 for p in action_1_probs),
                f"State {state_idx}: Action 1 probabilities varied across queries"
            )

            # Verify probabilities sum to 1
            self.assertAlmostEqual(
                action_0_probs[0] + action_1_probs[0],
                1.0,
                places=6,
                msg=f"State {state_idx}: Probabilities don't sum to 1"
            )

            print(f"✓ State {state_idx}: Action probabilities consistent: "
                  f"[{action_0_probs[0]:.6f}, {action_1_probs[0]:.6f}]")

    def test_probability_distribution_properties(self):
        """Test that probability distributions satisfy required properties."""
        for state_idx, state in enumerate(self.test_states):
            actions, probs = self.agent.model_checking_select_action(state, prob_threshold=0.0)

            # Test 1: All probabilities are non-negative
            self.assertTrue(
                np.all(probs >= 0),
                f"State {state_idx}: Found negative probabilities"
            )

            # Test 2: All probabilities are at most 1
            self.assertTrue(
                np.all(probs <= 1),
                f"State {state_idx}: Found probabilities > 1"
            )

            # Test 3: Probabilities sum to approximately 1
            # (Get full distribution)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                full_probs = self.agent.policy.forward(state_tensor).numpy()

            self.assertAlmostEqual(
                np.sum(full_probs),
                1.0,
                places=6,
                msg=f"State {state_idx}: Probabilities don't sum to 1"
            )

    def test_determinism_with_noise_disabled(self):
        """Test determinism when noise is explicitly disabled."""
        state = self.test_states[0]

        # Ensure noise is disabled
        self.assertFalse(self.agent.noise_enabled)
        self.assertFalse(self.agent.policy.noise_enabled)

        # Query 100 times
        probabilities = []
        for _ in range(100):
            actions, probs = self.agent.model_checking_select_action(state, prob_threshold=0.0)
            probabilities.append(probs.copy())

        # Verify perfect consistency
        reference = probabilities[0]
        for i, probs in enumerate(probabilities[1:], 1):
            max_diff = np.max(np.abs(probs - reference))
            self.assertLess(
                max_diff,
                1e-10,
                f"Query {i}: Maximum probability difference {max_diff} exceeds threshold"
            )

        print(f"✓ Noise disabled: All 100 queries identical (max diff < 1e-10)")

    def test_determinism_with_noise_enabled(self):
        """Test determinism when noise is enabled (critical for noisy model checking)."""
        # Enable depolarizing noise
        self.agent.enable_noise(depolarizing_p=0.01)

        self.assertTrue(self.agent.noise_enabled)
        self.assertTrue(self.agent.policy.noise_enabled)

        state = self.test_states[0]

        # Query 100 times with noise enabled
        probabilities = []
        for _ in range(100):
            actions, probs = self.agent.model_checking_select_action(state, prob_threshold=0.0)
            probabilities.append(probs.copy())

        # Verify perfect consistency even with noise
        reference = probabilities[0]
        for i, probs in enumerate(probabilities[1:], 1):
            max_diff = np.max(np.abs(probs - reference))
            self.assertLess(
                max_diff,
                1e-10,
                f"Query {i} with noise: Maximum probability difference {max_diff} exceeds threshold"
            )

        print(f"✓ Noise enabled (p=0.01): All 100 queries identical (max diff < 1e-10)")
        print(f"  Noisy probabilities: {reference}")

    def test_different_states_produce_different_distributions(self):
        """Test that different states produce different probability distributions."""
        distributions = []

        for state in self.test_states:
            actions, probs = self.agent.model_checking_select_action(state, prob_threshold=0.0)
            distributions.append(probs)

        # Verify that not all distributions are identical
        # (At least some pairs should be different)
        num_different = 0
        for i in range(len(distributions)):
            for j in range(i + 1, len(distributions)):
                if not np.allclose(distributions[i], distributions[j], atol=1e-6):
                    num_different += 1

        self.assertGreater(
            num_different,
            0,
            "All states produced identical distributions - policy may not be state-dependent"
        )

        print(f"✓ Found {num_different} distinct distribution pairs (out of {len(distributions)*(len(distributions)-1)//2})")

    def test_probability_threshold_filtering(self):
        """Test that probability threshold filtering is deterministic."""
        state = self.test_states[0]
        threshold = 0.3

        # Query with threshold 100 times
        filtered_results = []
        for _ in range(100):
            actions, probs = self.agent.model_checking_select_action(state, prob_threshold=threshold)
            filtered_results.append((actions.copy(), probs.copy()))

        # Verify all results are identical
        reference_actions, reference_probs = filtered_results[0]
        for i, (actions, probs) in enumerate(filtered_results[1:], 1):
            np.testing.assert_array_equal(
                actions,
                reference_actions,
                err_msg=f"Query {i}: Different actions selected with threshold {threshold}"
            )
            np.testing.assert_array_almost_equal(
                probs,
                reference_probs,
                decimal=10,
                err_msg=f"Query {i}: Different probabilities with threshold {threshold}"
            )

        print(f"✓ Threshold filtering (p={threshold}): All 100 queries identical")
        print(f"  Selected actions: {reference_actions}, probabilities: {reference_probs}")

    def test_quantum_circuit_reproducibility(self):
        """Test that the quantum circuit execution is reproducible."""
        state = self.test_states[0]
        state_tensor = torch.FloatTensor(state)

        # Execute circuit multiple times
        measurements_list = []
        for _ in range(50):
            with torch.no_grad():
                action_probs = self.agent.policy.forward(state_tensor)
                measurements_list.append(action_probs.numpy().copy())

        # Verify all measurements are identical
        reference = measurements_list[0]
        for i, measurements in enumerate(measurements_list[1:], 1):
            np.testing.assert_array_almost_equal(
                reference,
                measurements,
                decimal=10,
                err_msg=f"Circuit execution {i} produced different measurements"
            )

        print(f"✓ Quantum circuit: All 50 executions identical")


class TestQuantumPolicyNetworkDeterminism(unittest.TestCase):
    """Test deterministic behavior of quantum policy network directly."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)

        self.policy = QuantumPolicyNetwork(
            state_dim=4,
            action_dim=2,
            num_qubits=4,
            num_layers=3,
            use_parameter_shift=True
        )

    def test_amplitude_encoding_deterministic(self):
        """Test that amplitude encoding is deterministic."""
        state = np.array([0.5, 0.3, 0.2, 0.1])

        encodings = []
        for _ in range(100):
            encoded = self.policy._amplitude_encode_state(state)
            encodings.append(encoded.copy())

        # Verify all encodings are identical
        reference = encodings[0]
        for i, encoded in enumerate(encodings[1:], 1):
            np.testing.assert_array_almost_equal(
                reference,
                encoded,
                decimal=15,
                err_msg=f"Amplitude encoding {i} differs from reference"
            )

        # Verify normalization
        self.assertAlmostEqual(
            np.linalg.norm(reference),
            1.0,
            places=10,
            msg="Amplitude encoding not properly normalized"
        )

    def test_act_method_stochastic_but_distribution_deterministic(self):
        """Test that act() samples stochastically but from a deterministic distribution."""
        state = torch.FloatTensor([0.5, 0.3, 0.2, 0.1])

        # Get action probabilities (should be deterministic)
        distributions = []
        for _ in range(50):
            with torch.no_grad():
                action_probs = self.policy.forward(state)
                distributions.append(action_probs.numpy().copy())

        # Verify all distributions are identical
        reference = distributions[0]
        for i, dist in enumerate(distributions[1:], 1):
            np.testing.assert_array_almost_equal(
                reference,
                dist,
                decimal=10,
                err_msg=f"Distribution {i} differs from reference"
            )

        print(f"✓ Policy produces deterministic distribution: {reference}")

        # The actual actions sampled will vary, but the distribution should not
        sampled_actions = []
        for _ in range(100):
            action, log_prob, greedy_action = self.policy.act(state)
            sampled_actions.append(action)

        # Should see both actions sampled (stochastic sampling)
        unique_actions = set(sampled_actions)
        self.assertGreater(
            len(unique_actions),
            1,
            "act() appears to be deterministic - should sample stochastically"
        )

        print(f"✓ Sampled {len(unique_actions)} unique actions from {len(sampled_actions)} samples (stochastic)")


def run_tests():
    """Run all tests with verbose output."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestQuantumReinforceDeterminism))
    suite.addTests(loader.loadTestsFromTestCase(TestQuantumPolicyNetworkDeterminism))

    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
