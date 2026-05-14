"""Hoeffding transition updater for black-box policies.

For i.i.d. samples ``X_1, ..., X_N`` in ``[0, 1]`` with empirical mean
``p_hat`` and per-interval failure budget ``delta``, the Hoeffding
half-width is

    r_N(delta) = sqrt( log(2 / delta) / (2 N) ),
    pi^- = max(0, p_hat - r_N(delta)),
    pi^+ = min(1, p_hat + r_N(delta)).

This is the bound used in the paper's methodology section. It is
distribution-free (does not depend on ``p_hat``), so the resulting
interval is symmetric and equally wide at every estimate.

Sample-size inversion from a target half-width ``h`` is

    N = ceil( log(2 / delta) / (2 h^2) ).

The two-level Bonferroni correction, greedy propagation through the
known MDP transitions, skippable-state detection, graph-preservation
clamp, and IDTMC construction are inherited unchanged from
:class:`PolicySamplingTransitionUpdater`.

Config string:
    ``hoeffding;alpha=0.05;samples=500;half_width=0.05;seed=0;num_models=1``
"""
import math

from common.transition_updaters.policy_sampling_transition_updater import (
    PolicySamplingTransitionUpdater,
)


class HoeffdingTransitionUpdater(PolicySamplingTransitionUpdater):

    METHOD_NAME: str = "Hoeffding (i.i.d.)"
    METRIC_PREFIX: str = "ho"

    def get_updater_name(self) -> str:
        return (f"hoeffding(alpha={self.alpha},"
                f"samples={self.samples},half_width={self.half_width})")

    # ---- CI hooks ----

    @staticmethod
    def _hoeffding_radius(alpha_per: float, samples: int) -> float:
        if samples <= 0:
            return float("inf")
        return math.sqrt(math.log(2.0 / alpha_per) / (2.0 * samples))

    def _ci_radius(self, alpha_per: float, samples: int) -> float:
        return self._hoeffding_radius(alpha_per, samples)

    def _ci_samples_from_half_width(self, alpha_per: float,
                                    half_width: float) -> int:
        return int(math.ceil(
            math.log(2.0 / alpha_per) / (2.0 * half_width ** 2)))

    def _ci_bounds(self, n_a: int, samples: int, alpha_per: float) -> tuple:
        if samples == 0:
            return 0.0, 1.0
        p_hat = n_a / samples
        r = self._hoeffding_radius(alpha_per, samples)
        return max(0.0, p_hat - r), min(1.0, p_hat + r)

    def _banner_ci_lines(self, alpha_per: float, samples: int) -> list:
        r = self._hoeffding_radius(alpha_per, samples)
        return [f"  Hoeffding radius       : {r:.4f}"]

    def _mlflow_ci_params(self, alpha_per: float, samples: int) -> dict:
        return {f"{self.METRIC_PREFIX}_radius":
                self._hoeffding_radius(alpha_per, samples)}
