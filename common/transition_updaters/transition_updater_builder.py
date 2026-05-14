from typing import List, Optional
from common.transition_updaters.transition_updater import TransitionUpdater
from common.transition_updaters.epsilon_transition_updater import EpsilonTransitionUpdater
from common.transition_updaters.policy_sampling_transition_updater import PolicySamplingTransitionUpdater
from common.transition_updaters.hoeffding_transition_updater import HoeffdingTransitionUpdater
from common.transition_updaters.empirical_bernstein_transition_updater import EmpiricalBernsteinTransitionUpdater


class TransitionUpdaterBuilder:
    """Factory class for building transition updaters from configuration strings.

    Updaters can be chained using '#' separator, same as preprocessors/labelers.
    Each updater config has ';' separated fields:
        <name>;<key=value>;<key=value>...

    The PCTL property is specified via --prop. For interval models, the model
    checker automatically checks both Pmax=? and Pmin=?.

    Examples:
        "epsilon;eps=0.05"
        "hoeffding;alpha=0.05;samples=500;seed=0"
        "empirical_bernstein;alpha=0.05;samples=500;seed=0"
    """

    @staticmethod
    def build_transition_updaters(config_str: str) -> Optional[List[TransitionUpdater]]:
        """Build transition updaters from a configuration string.

        Args:
            config_str: '#'-separated updater configs, or '' / 'None' for no updaters.

        Returns:
            List of TransitionUpdater instances, or None if none configured.
        """
        if config_str == "" or config_str == "None":
            return None

        updaters = []
        for updater_str in config_str.split("#"):
            name = updater_str.split(";")[0].strip()
            if name == "epsilon":
                updaters.append(EpsilonTransitionUpdater(updater_str))
            elif name == "policy_sampling":
                updaters.append(PolicySamplingTransitionUpdater(updater_str))
            elif name == "hoeffding":
                updaters.append(HoeffdingTransitionUpdater(updater_str))
            elif name == "empirical_bernstein":
                updaters.append(EmpiricalBernsteinTransitionUpdater(updater_str))
            elif name != "":
                print(f"Warning: Unknown transition updater '{name}'")

        return updaters if updaters else None
