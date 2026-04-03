"""Remote Storm/stormpy client.

Exposes Storm 1.12.0 model-checking capabilities through the COOL-MC server
without requiring a local Storm or stormpy installation.

Obtain via ``mc.storm`` after creating a :class:`~coolmc.client.CoolMC` instance.

Example::

    mc = coolmc.CoolMC()

    # Model info
    info = mc.storm.build_model("transporter.prism", constants="MAX_JOBS=2,MAX_FUEL=10")
    print(info.nr_states, info.labels)

    # Model checking
    res = mc.storm.check("transporter.prism", "Pmax=? [ F jobs_done=2 ]",
                         constants="MAX_JOBS=2,MAX_FUEL=10")
    print(float(res))  # e.g. 0.75

    # Optimal scheduler
    sched = mc.storm.get_scheduler("transporter.prism", "Pmax=? [ F jobs_done=2 ]",
                                   constants="MAX_JOBS=2,MAX_FUEL=10")
    print(sched.get_action(0))  # action chosen in state 0

    # Simulation
    sim = mc.storm.simulate("transporter.prism", nr_steps=20,
                            constants="MAX_JOBS=2,MAX_FUEL=10", seed=42)
    for step in sim.path:
        print(step)

    # Parametric model checking
    pres = mc.storm.parametric_check("die.prism", "P=? [ F s=7 & d=2 ]")
    print(pres.result)   # rational function string

    # Full transition matrix
    trans = mc.storm.get_transitions("transporter.prism",
                                     constants="MAX_JOBS=2,MAX_FUEL=10")
    for t in trans["transitions"][:5]:
        print(t)
"""
from __future__ import annotations

from typing import Any, Optional

import requests


# ---------------------------------------------------------------------------
# Result wrapper types
# ---------------------------------------------------------------------------

class ModelInfo:
    """Statistics returned by :meth:`Storm.build_model`."""

    def __init__(self, data: dict) -> None:
        self._data = data

    @property
    def model_type(self) -> str:
        """Storm model type string, e.g. ``'ModelType.MDP'``."""
        return self._data["model_type"]

    @property
    def nr_states(self) -> int:
        return self._data["nr_states"]

    @property
    def nr_transitions(self) -> int:
        return self._data["nr_transitions"]

    @property
    def labels(self) -> list[str]:
        """Atomic proposition labels present in the model."""
        return self._data["labels"]

    @property
    def reward_models(self) -> list[str]:
        return self._data["reward_models"]

    @property
    def initial_states(self) -> list[int]:
        return self._data["initial_states"]

    def __repr__(self) -> str:
        return (
            f"<ModelInfo type={self.model_type} states={self.nr_states} "
            f"transitions={self.nr_transitions} labels={self.labels}>"
        )


class StormResult:
    """Result of a model-checking call via :meth:`Storm.check`."""

    def __init__(self, data: dict) -> None:
        self._data = data

    @property
    def result(self) -> float:
        """Numeric result for the initial state."""
        return self._data["result"]

    @property
    def initial_state(self) -> int:
        return self._data.get("initial_state", 0)

    @property
    def model(self) -> ModelInfo:
        return ModelInfo(self._data.get("model", {}))

    @property
    def all_states(self) -> Optional[list[float]]:
        """Per-state results (only populated when ``all_states=True``)."""
        return self._data.get("all_states")

    def __float__(self) -> float:
        return float(self.result)

    def __repr__(self) -> str:
        return f"<StormResult result={self.result} states={self.model.nr_states}>"


class SchedulerResult:
    """Result of a scheduler-extraction call via :meth:`Storm.get_scheduler`."""

    def __init__(self, data: dict) -> None:
        self._data = data

    @property
    def result(self) -> float:
        """Optimal value for the initial state."""
        return self._data["result"]

    @property
    def choices(self) -> dict[str, int]:
        """Maps state ID (``str``) to the chosen action index (``int``)."""
        return self._data["choices"]

    @property
    def model(self) -> ModelInfo:
        return ModelInfo(self._data.get("model", {}))

    def get_action(self, state_id: int) -> Optional[int]:
        """Return the optimal action index for *state_id*, or ``None``."""
        return self.choices.get(str(state_id))

    def __repr__(self) -> str:
        return f"<SchedulerResult result={self.result} nr_states={len(self.choices)}>"


class SimulationResult:
    """Result of a simulation run via :meth:`Storm.simulate`."""

    def __init__(self, data: dict) -> None:
        self._data = data

    @property
    def path(self) -> list[dict]:
        """List of steps: ``[{step, state, labels, action}, ...]``."""
        return self._data["path"]

    @property
    def final_state(self) -> Any:
        return self._data["final_state"]

    @property
    def is_done(self) -> bool:
        """``True`` if the simulator reached a terminal state."""
        return self._data["is_done"]

    @property
    def nr_steps_taken(self) -> int:
        return self._data["nr_steps_taken"]

    def __repr__(self) -> str:
        return (
            f"<SimulationResult steps={self.nr_steps_taken} "
            f"final_state={self.final_state} done={self.is_done}>"
        )


class IntervalResult:
    """Result of interval model checking via :meth:`Storm.interval_check`."""

    def __init__(self, data: dict) -> None:
        self._data = data

    @property
    def min_result(self) -> float:
        """Lower bound (minimizing uncertainty resolution)."""
        return self._data["min_result"]

    @property
    def max_result(self) -> float:
        """Upper bound (maximizing uncertainty resolution)."""
        return self._data["max_result"]

    @property
    def eps(self) -> float:
        """Epsilon perturbation used."""
        return self._data["eps"]

    @property
    def model(self) -> ModelInfo:
        return ModelInfo(self._data.get("model", {}))

    def __repr__(self) -> str:
        return f"<IntervalResult min={self.min_result} max={self.max_result} eps={self.eps}>"


class ParametricResult:
    """Result of parametric model checking via :meth:`Storm.parametric_check`."""

    def __init__(self, data: dict) -> None:
        self._data = data

    @property
    def result(self) -> str:
        """Rational function as a string, e.g. ``'1/6'`` or ``'p/(1-p+p**2)'``."""
        return self._data["result"]

    @property
    def parameters(self) -> list[str]:
        """Names of the free parameters in the model."""
        return self._data["parameters"]

    @property
    def model(self) -> ModelInfo:
        return ModelInfo(self._data.get("model", {}))

    def __repr__(self) -> str:
        return f"<ParametricResult result={self.result!r} parameters={self.parameters}>"


# ---------------------------------------------------------------------------
# Main client class
# ---------------------------------------------------------------------------

class Storm:
    """Remote interface to Storm 1.12.0 running inside the COOL-MC Docker container.

    All methods mirror the stormpy API but execute server-side — no local
    Storm installation required.
    """

    def __init__(self, server_url: str) -> None:
        self._url = server_url.rstrip("/")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, endpoint: str, payload: dict) -> dict:
        resp = requests.post(f"{self._url}{endpoint}", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()

    def _req(
        self,
        prism_file_path: str,
        formula: str = "",
        constants: str = "",
        **extra,
    ) -> dict:
        return {
            "prism_file_path": prism_file_path,
            "constant_definitions": constants,
            "formula": formula,
            **extra,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_model(
        self,
        prism_file_path: str,
        *,
        constants: str = "",
        formula: str = "",
    ) -> ModelInfo:
        """Build a stormpy model and return its statistics.

        Args:
            prism_file_path: Path to the PRISM file (relative or absolute).
            constants:        ``'KEY=VAL,KEY2=VAL2'`` constant definitions.
            formula:          Optional formula — building with a property enables
                              formula-directed state-space exploration.

        Returns a :class:`ModelInfo` with ``nr_states``, ``nr_transitions``,
        ``labels``, etc.
        """
        data = self._post(
            "/storm/build-model",
            self._req(prism_file_path, formula=formula, constants=constants),
        )
        return ModelInfo(data)

    def check(
        self,
        prism_file_path: str,
        formula: str,
        *,
        constants: str = "",
        all_states: bool = False,
    ) -> StormResult:
        """Model-check a property and return the result for the initial state.

        Args:
            prism_file_path: PRISM file to analyse.
            formula:         PCTL/LTL/reward formula, e.g. ``"Pmax=? [ F done ]"``.
            constants:       Constant definitions string.
            all_states:      Also return results for every state (``result.all_states``).

        Returns a :class:`StormResult`. Use ``float(result)`` for the numeric value.
        """
        data = self._post(
            "/storm/check",
            self._req(
                prism_file_path,
                formula=formula,
                constants=constants,
                all_states=all_states,
            ),
        )
        return StormResult(data)

    def get_scheduler(
        self,
        prism_file_path: str,
        formula: str,
        *,
        constants: str = "",
    ) -> SchedulerResult:
        """Compute an optimal scheduler and return per-state action choices.

        Args:
            prism_file_path: PRISM file.
            formula:         Optimisation formula, e.g. ``"Pmax=? [ F done ]"``.
            constants:       Constant definitions string.

        Returns a :class:`SchedulerResult`. Use ``result.get_action(state_id)`` to
        look up the optimal action for a given state.
        """
        data = self._post(
            "/storm/get-scheduler",
            self._req(prism_file_path, formula=formula, constants=constants),
        )
        return SchedulerResult(data)

    def simulate(
        self,
        prism_file_path: str,
        *,
        nr_steps: int = 100,
        constants: str = "",
        seed: Optional[int] = None,
    ) -> SimulationResult:
        """Run a Monte Carlo simulation and return the execution trace.

        The simulator selects the first available action at each step.

        Args:
            prism_file_path: PRISM file.
            nr_steps:        Maximum number of simulation steps.
            constants:       Constant definitions string.
            seed:            Integer seed for reproducible traces.

        Returns a :class:`SimulationResult` with ``path``, ``final_state``, and
        ``is_done``.
        """
        data = self._post(
            "/storm/simulate",
            self._req(
                prism_file_path,
                constants=constants,
                nr_steps=nr_steps,
                seed=seed,
            ),
        )
        return SimulationResult(data)

    def parametric_check(
        self,
        prism_file_path: str,
        formula: str,
        *,
        constants: str = "",
    ) -> ParametricResult:
        """Perform parametric model checking and return the result as a rational function.

        Use this with models that contain uninstantiated probability/rate parameters.

        Args:
            prism_file_path: PRISM file with parametric transitions.
            formula:         Property formula.
            constants:       Any additional constant definitions.

        Returns a :class:`ParametricResult` with ``result`` (rational function string)
        and ``parameters`` (list of parameter names).
        """
        data = self._post(
            "/storm/parametric-check",
            self._req(prism_file_path, formula=formula, constants=constants),
        )
        return ParametricResult(data)

    def interval_check(
        self,
        prism_file_path: str,
        formula: str,
        *,
        constants: str = "",
        eps: float = 0.05,
    ) -> IntervalResult:
        """Build an interval model and check with both min and max bounds.

        Each transition probability *p* is widened to
        ``[max(0, p - eps), min(1, p + eps)]``, then Storm checks the
        property under both minimizing and maximizing uncertainty resolution.

        Args:
            prism_file_path: PRISM file to analyse.
            formula:         PCTL formula, e.g. ``"Pmax=? [ F done ]"``.
            constants:       Constant definitions string.
            eps:             Epsilon perturbation (default 0.05).

        Returns an :class:`IntervalResult` with ``min_result`` and ``max_result``.

        Example::

            res = mc.storm.interval_check(
                "transporter.prism",
                "Pmax=? [ F jobs_done=2 ]",
                constants="MAX_JOBS=2,MAX_FUEL=10",
                eps=0.1,
            )
            print(res.min_result, res.max_result)  # e.g. 0.85, 1.0
        """
        data = self._post(
            "/storm/interval-check",
            self._req(
                prism_file_path,
                formula=formula,
                constants=constants,
                eps=eps,
            ),
        )
        return IntervalResult(data)

    def get_transitions(
        self,
        prism_file_path: str,
        *,
        constants: str = "",
    ) -> dict:
        """Return the full sparse transition matrix as a list of dicts.

        Each entry has keys ``from``, ``action``, ``to``, and ``probability``.
        The response dict also includes model statistics (``model_type``,
        ``nr_states``, ``nr_transitions``).

        Args:
            prism_file_path: PRISM file.
            constants:       Constant definitions string.
        """
        return self._post(
            "/storm/transitions",
            self._req(prism_file_path, constants=constants),
        )
