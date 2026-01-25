import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import common.agents first to ensure it's available for model_checker type hints
import common.agents
from common.safe_gym.action_mapper import ActionMapper
from common.safe_gym.storm_bridge import StormBridge


# Configuration for all PRISM files with their required constants
PRISM_FILES_CONFIG = {
    "transporter.prism": "MAX_JOBS=2,MAX_FUEL=10",
    "avoid.prism": "xMax=4,yMax=4,slickness=0.1",
    "frozen_lake.prism": "start_position=0,control=0.333",
    "resource_gathering.prism": "B=50,GOLD_TO_COLLECT=5,GEM_TO_COLLECT=5",
    "zeroconf.prism": "N=20,K=4,reset=true",
    "csma.2-2.v1.prism": "",  # All constants have defaults
    "navigation.prism": "",  # All constants have defaults
    "pacman.prism": "MAXSTEPS=5",
    "scheduling_task.prism": "",  # All constants have defaults
}


class TestActionMapper(unittest.TestCase):

    def setUp(self):
        self.prism_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'prism_files'
        ))
        self.wrong_action_penalty = 1000
        self.reward_flag = False
        self.disabled_features = ""
        self.seed = 42

    def _test_extraction_vs_simulation(self, prism_filename, constant_definitions):
        """Helper method to test extraction vs simulation for a given PRISM file."""
        prism_file_path = os.path.join(self.prism_dir, prism_filename)

        if not os.path.exists(prism_file_path):
            self.skipTest(f"PRISM file not found: {prism_file_path}")

        # Create storm bridge
        storm_bridge = StormBridge(
            prism_file_path,
            constant_definitions,
            self.wrong_action_penalty,
            self.reward_flag,
            self.disabled_features,
            self.seed
        )

        # Collect actions using analytical approach
        analytical_mapper = ActionMapper.collect_actions(storm_bridge, analytical=True)

        # Collect actions using simulation-based approach
        simulation_mapper = ActionMapper.collect_actions(storm_bridge, analytical=False)

        # Both should have the same actions (sorted)
        self.assertEqual(
            analytical_mapper.actions,
            simulation_mapper.actions,
            f"[{prism_filename}] Analytical actions {analytical_mapper.actions} != "
            f"Simulation actions {simulation_mapper.actions}"
        )

        # Verify action count matches
        self.assertEqual(
            analytical_mapper.get_action_count(),
            simulation_mapper.get_action_count(),
            f"[{prism_filename}] Action counts don't match"
        )

        return analytical_mapper.actions

    def test_transporter(self):
        """Test transporter.prism"""
        actions = self._test_extraction_vs_simulation(
            "transporter.prism",
            PRISM_FILES_CONFIG["transporter.prism"]
        )
        self.assertGreater(len(actions), 0)

    def test_avoid(self):
        """Test avoid.prism"""
        actions = self._test_extraction_vs_simulation(
            "avoid.prism",
            PRISM_FILES_CONFIG["avoid.prism"]
        )
        self.assertGreater(len(actions), 0)

    def test_frozen_lake(self):
        """Test frozen_lake.prism"""
        actions = self._test_extraction_vs_simulation(
            "frozen_lake.prism",
            PRISM_FILES_CONFIG["frozen_lake.prism"]
        )
        self.assertGreater(len(actions), 0)

    def test_resource_gathering(self):
        """Test resource_gathering.prism"""
        actions = self._test_extraction_vs_simulation(
            "resource_gathering.prism",
            PRISM_FILES_CONFIG["resource_gathering.prism"]
        )
        self.assertGreater(len(actions), 0)

    def test_zeroconf(self):
        """Test zeroconf.prism

        Note: zeroconf has many states that are hard to reach via random simulation.
        The analytical approach finds all 14 actions, while simulation may only find ~7.
        This test verifies that analytical finds a superset of simulation actions.
        """
        prism_file_path = os.path.join(self.prism_dir, "zeroconf.prism")
        constant_definitions = PRISM_FILES_CONFIG["zeroconf.prism"]

        storm_bridge = StormBridge(
            prism_file_path,
            constant_definitions,
            self.wrong_action_penalty,
            self.reward_flag,
            self.disabled_features,
            self.seed
        )

        analytical_mapper = ActionMapper.collect_actions(storm_bridge, analytical=True)
        simulation_mapper = ActionMapper.collect_actions(storm_bridge, analytical=False)

        print(f"\n[zeroconf] Analytical actions ({analytical_mapper.get_action_count()}): {analytical_mapper.actions}")
        print(f"[zeroconf] Simulation actions ({simulation_mapper.get_action_count()}): {simulation_mapper.actions}")

        # Analytical should find at least as many actions as simulation
        self.assertGreaterEqual(
            analytical_mapper.get_action_count(),
            simulation_mapper.get_action_count(),
            "Analytical should find at least as many actions as simulation"
        )

        # All simulation actions should be in analytical actions (superset check)
        for action in simulation_mapper.actions:
            self.assertIn(
                action,
                analytical_mapper.actions,
                f"Simulation action '{action}' not found in analytical actions"
            )

    def test_csma(self):
        """Test csma.2-2.v1.prism"""
        actions = self._test_extraction_vs_simulation(
            "csma.2-2.v1.prism",
            PRISM_FILES_CONFIG["csma.2-2.v1.prism"]
        )
        self.assertGreater(len(actions), 0)

    def test_navigation(self):
        """Test navigation.prism"""
        actions = self._test_extraction_vs_simulation(
            "navigation.prism",
            PRISM_FILES_CONFIG["navigation.prism"]
        )
        self.assertGreater(len(actions), 0)

    def test_pacman(self):
        """Test pacman.prism"""
        actions = self._test_extraction_vs_simulation(
            "pacman.prism",
            PRISM_FILES_CONFIG["pacman.prism"]
        )
        self.assertGreater(len(actions), 0)

    def test_scheduling_task(self):
        """Test scheduling_task.prism"""
        actions = self._test_extraction_vs_simulation(
            "scheduling_task.prism",
            PRISM_FILES_CONFIG["scheduling_task.prism"]
        )
        self.assertGreater(len(actions), 0)

    def test_extraction_collect_actions_directly(self):
        """Test that collect_actions_by_extraction works directly with path."""
        prism_file_path = os.path.join(self.prism_dir, "transporter.prism")
        analytical_mapper = ActionMapper.collect_actions_by_extraction(
            prism_file_path,
            PRISM_FILES_CONFIG["transporter.prism"]
        )

        # Should have at least one action
        self.assertGreater(analytical_mapper.get_action_count(), 0)

        # Actions should be sorted
        self.assertEqual(analytical_mapper.actions, sorted(analytical_mapper.actions))


if __name__ == '__main__':
    unittest.main()
