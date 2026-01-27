from typing import List, Optional
from common.state_labelers.state_labeler import StateLabeler
from common.state_labelers.critical_state_labeler import CriticalStateLabeler
from common.state_labelers.top_two_gap_labeler import TopTwoGapLabeler


class StateLabelerBuilder:
    """Factory class for building state labelers from configuration strings.

    State labelers can be chained using '#' separator, similar to preprocessors.
    Each labeler has its own configuration with ';' separated key=value pairs.

    Example configurations:
        - "critical_state;min=0.3;max=0.7"
        - "critical_state;min=0.2;max=0.8#another_labeler;param=value"
    """

    @staticmethod
    def build_state_labelers(config_str: str) -> Optional[List[StateLabeler]]:
        """Build state labelers from a configuration string.

        Args:
            config_str: Configuration string with '#' separated labeler configs.
                        Each labeler config has ';' separated parameters.
                        Empty string or "None" returns None.

        Returns:
            List of StateLabeler instances, or None if no labelers configured.
        """
        if config_str == "" or config_str == "None":
            return None

        labelers = []

        for labeler_str in config_str.split("#"):
            labeler_name = labeler_str.split(";")[0].strip()

            if labeler_name == "critical_state":
                labeler = CriticalStateLabeler(labeler_str)
                labelers.append(labeler)
            elif labeler_name == "top_two_gap":
                labeler = TopTwoGapLabeler(labeler_str)
                labelers.append(labeler)
            elif labeler_name != "":
                print(f"Warning: Unknown state labeler '{labeler_name}'")

        return labelers if labelers else None
