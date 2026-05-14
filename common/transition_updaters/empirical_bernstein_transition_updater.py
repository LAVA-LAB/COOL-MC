"""Empirical Bernstein transition updater for black-box policies.

Implements the i.i.d. empirical Bernstein bound of Maurer & Pontil (2009),
matching the paper's methodology section. For i.i.d. samples
``X_1, ..., X_N`` in ``[0, 1]`` with empirical mean ``p_hat`` and
Bessel-corrected sample variance

    sigma_hat^2 = (1 / (N - 1)) * sum_i (X_i - p_hat)^2,

the half-width at per-interval failure budget ``delta`` is

    r_N(delta) = sqrt( 2 * sigma_hat^2 * log(4 / delta) / N )
               + 7 * log(4 / delta) / (3 * (N - 1)).

For Bernoulli counts (a single action's votes), the sample variance
simplifies to

    sigma_hat^2 = (N / (N - 1)) * p_hat * (1 - p_hat),

so the variance term becomes ``sqrt(2 * p_hat * (1 - p_hat)
* log(4 / delta) / (N - 1))``. The bound is variance-adaptive: it
shrinks toward ``7 * log(4 / delta) / (3 * (N - 1))`` as ``p_hat -> 0``
or ``p_hat -> 1``, which is much tighter than the distribution-free
Hoeffding radius for skewed action distributions. The bound is only
defined for ``N >= 2``.

The two-level Bonferroni correction, greedy propagation through the
known MDP transitions, skippable-state detection, graph-preservation
clamp, and IDTMC construction are inherited unchanged from
:class:`PolicySamplingTransitionUpdater`.

Config string:
    ``empirical_bernstein;alpha=0.05;samples=500;half_width=0.05;seed=0;num_models=1``
"""
import math

from common.transition_updaters.policy_sampling_transition_updater import (
    PolicySamplingTransitionUpdater,
)


class EmpiricalBernsteinTransitionUpdater(PolicySamplingTransitionUpdater):

    METHOD_NAME: str = "Empirical-Bernstein (Maurer-Pontil 2009, i.i.d.)"
    METRIC_PREFIX: str = "eb"

    def get_updater_name(self) -> str:
        return (f"empirical_bernstein(alpha={self.alpha},"
                f"samples={self.samples},half_width={self.half_width})")

    # ---- CI hooks ----

    @staticmethod
    def _eb_radius(p_hat: float, n: int, alpha_per: float) -> float:
        """Maurer-Pontil 2009 empirical Bernstein half-width for i.i.d.
        Bernoulli samples.

            r_N = sqrt( 2 * sigma_hat^2 * log(4 / alpha_per) / N )
                + 7 * log(4 / alpha_per) / (3 * (N - 1)),

        with the Bessel-corrected Bernoulli variance
        ``sigma_hat^2 = (N / (N - 1)) * p_hat * (1 - p_hat)``. Plugging
        this in collapses the variance term to
        ``sqrt(2 * p_hat * (1 - p_hat) * log(4 / alpha_per) / (N - 1))``.
        """
        if n < 2:
            return 1.0
        u = math.log(4.0 / alpha_per)
        variance_term = math.sqrt(
            2.0 * p_hat * (1.0 - p_hat) * u / (n - 1)
        )
        deterministic_term = 7.0 * u / (3.0 * (n - 1))
        return variance_term + deterministic_term

    @classmethod
    def _eb_worst_case_radius(cls, alpha_per: float, n: int) -> float:
        """Worst-case (p_hat = 0.5) half-width used for sample-size
        planning."""
        return cls._eb_radius(0.5, n, alpha_per)

    def _ci_radius(self, alpha_per: float, samples: int) -> float:
        return self._eb_worst_case_radius(alpha_per, samples)

    def _ci_samples_from_half_width(self, alpha_per: float,
                                    half_width: float) -> int:
        """Invert the worst-case radius via doubling + bisection."""
        lo, hi = 2, 8
        while self._eb_worst_case_radius(alpha_per, hi) > half_width:
            hi *= 2
            if hi > 10 ** 9:
                return hi
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if self._eb_worst_case_radius(alpha_per, mid) > half_width:
                lo = mid
            else:
                hi = mid
        return hi

    def _ci_bounds(self, n_a: int, samples: int, alpha_per: float) -> tuple:
        if samples < 2:
            return 0.0, 1.0
        p_hat = n_a / samples
        r = self._eb_radius(p_hat, samples, alpha_per)
        return max(0.0, p_hat - r), min(1.0, p_hat + r)

    def _banner_ci_lines(self, alpha_per: float, samples: int) -> list:
        if samples < 2:
            return ["  EB requires N >= 2 (current N too small)"]
        u = math.log(4.0 / alpha_per)
        det_limit = 7.0 * u / (3.0 * (samples - 1))
        worst = self._eb_worst_case_radius(alpha_per, samples)
        return [
            f"  EB worst-case radius   : {worst:.4f}  (p_hat = 0.5)",
            f"  EB deterministic limit : {det_limit:.4f}  "
            "(p_hat in {0, 1})",
        ]

    def _mlflow_ci_params(self, alpha_per: float, samples: int) -> dict:
        if samples < 2:
            det_limit = float("inf")
        else:
            u = math.log(4.0 / alpha_per)
            det_limit = 7.0 * u / (3.0 * (samples - 1))
        return {
            f"{self.METRIC_PREFIX}_worst_case_radius":
                self._eb_worst_case_radius(alpha_per, samples),
            f"{self.METRIC_PREFIX}_deterministic_limit": det_limit,
        }
