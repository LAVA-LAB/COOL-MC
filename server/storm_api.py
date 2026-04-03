"""Server-side stormpy wrapper.

All stormpy calls run here — inside the Docker container where Storm 1.12.0
and the corresponding stormpy bindings are installed.  Clients call these
functions indirectly via the FastAPI endpoints in server.py and receive
JSON-serialisable results without needing a local Storm installation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import stormpy
import stormpy.simulator

WORKDIR = Path("/workspaces/coolmc")
_PRISM_USER_DIR = WORKDIR / "prism_files_user"
_PRISM_BUILTIN_DIR = WORKDIR / "prism_files"


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_prism_path(prism_file_path: str) -> Path:
    """Resolve a PRISM file path argument to an absolute Path.

    Accepted formats
    ----------------
    - Absolute path                   — used as-is.
    - ``prism_files_user/foo.prism``  — relative to WORKDIR.
    - Bare filename ``foo.prism``     — searched in prism_files_user/ then
                                        prism_files/ (built-in environments).
    """
    p = Path(prism_file_path)
    if p.is_absolute():
        if p.exists():
            return p
        raise FileNotFoundError(f"PRISM file not found: {p}")

    # Path relative to WORKDIR (e.g. "prism_files_user/foo.prism")
    rel = WORKDIR / p
    if rel.exists():
        return rel

    # Bare filename — search user dir, then built-in dir
    if "/" not in prism_file_path:
        user = _PRISM_USER_DIR / p.name
        if user.exists():
            return user
        builtin = _PRISM_BUILTIN_DIR / p.name
        if builtin.exists():
            return builtin

    raise FileNotFoundError(f"PRISM file not found: {prism_file_path!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_constants(program, constant_definitions: str):
    """Apply a ``'KEY=VAL,KEY2=VAL2'`` string to a PRISM program and return
    the updated program with those constants defined."""
    if not constant_definitions or not constant_definitions.strip():
        return program
    return stormpy.preprocess_symbolic_input(
        program, [], constant_definitions
    )[0].as_prism_program()


def _model_to_dict(model) -> dict:
    """Serialise core model statistics to a JSON-friendly dict."""
    return {
        "model_type": str(model.model_type),
        "nr_states": int(model.nr_states),
        "nr_transitions": int(model.nr_transitions),
        "labels": sorted(model.labeling.get_labels()),
        "reward_models": list(model.reward_models.keys()),
        "initial_states": [int(s) for s in model.initial_states],
    }


def _to_python(value) -> Any:
    """Convert a stormpy result value to a plain Python float or string."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Public API  (called by server.py FastAPI endpoints)
# ---------------------------------------------------------------------------

def build_model(
    prism_file_path: str,
    constant_definitions: str = "",
    formula: str = "",
) -> dict:
    """Parse and build a stormpy model; return model statistics.

    If *formula* is given the model is built with that property in scope
    (enabling formula-directed state-space exploration / bisimulation).
    """
    path = resolve_prism_path(prism_file_path)
    program = stormpy.parse_prism_program(str(path))
    program = _apply_constants(program, constant_definitions)

    if formula.strip():
        properties = stormpy.parse_properties(formula, program)
        model = stormpy.build_model(program, properties)
    else:
        model = stormpy.build_model(program)

    return _model_to_dict(model)


def check(
    prism_file_path: str,
    formula: str,
    constant_definitions: str = "",
    all_states: bool = False,
) -> dict:
    """Model-check a formula; return the result for the initial state.

    If *all_states* is ``True`` the response also contains a list with the
    result for every state in the model.
    """
    path = resolve_prism_path(prism_file_path)
    program = stormpy.parse_prism_program(str(path))
    program = _apply_constants(program, constant_definitions)

    properties = stormpy.parse_properties(formula, program)
    model = stormpy.build_model(program, properties)
    result = stormpy.model_checking(model, properties[0])

    initial_state = int(model.initial_states[0])
    response: dict = {
        "result": _to_python(result.at(initial_state)),
        "initial_state": initial_state,
        "model": _model_to_dict(model),
        "all_states": None,
    }

    if all_states:
        response["all_states"] = [
            _to_python(result.at(s)) for s in range(model.nr_states)
        ]

    return response


def get_scheduler(
    prism_file_path: str,
    formula: str,
    constant_definitions: str = "",
) -> dict:
    """Run model checking with scheduler extraction.

    Returns the optimal action choice for every reachable state.
    """
    path = resolve_prism_path(prism_file_path)
    program = stormpy.parse_prism_program(str(path))
    program = _apply_constants(program, constant_definitions)

    properties = stormpy.parse_properties(formula, program)
    model = stormpy.build_model(program, properties)

    env = stormpy.Environment()
    result = stormpy.model_checking(
        model, properties[0], environment=env, extract_scheduler=True
    )

    if not result.has_scheduler:
        raise ValueError("No scheduler available for this formula/model combination")

    scheduler = result.scheduler
    choices: dict[str, int] = {}
    for state in model.states:
        choice = scheduler.get_choice(state)
        choices[str(state.id)] = int(choice.get_deterministic_choice())

    initial_state = int(model.initial_states[0])
    return {
        "result": _to_python(result.at(initial_state)),
        "choices": choices,
        "model": _model_to_dict(model),
    }


def simulate(
    prism_file_path: str,
    nr_steps: int,
    constant_definitions: str = "",
    seed: Optional[int] = None,
) -> dict:
    """Run a Monte Carlo simulation and return the path as a list of step dicts.

    The simulator selects the first available action (alphabetically) at each
    step.  Use *seed* for reproducible results.

    Each step dict has keys ``step``, ``state`` (variable assignments as dict),
    and ``action`` (action name string, or ``None`` for the initial state).
    """
    import json

    path = resolve_prism_path(prism_file_path)
    program = stormpy.parse_prism_program(str(path))
    program = _apply_constants(program, constant_definitions)

    # Label any unlabelled commands so action-name mode works correctly.
    suggestions: dict = {}
    i = 0
    for module in program.modules:
        for command in module.commands:
            if not command.is_labeled:
                suggestions[command.global_index] = f"tau_{i}"
                i += 1
    program = program.label_unlabelled_commands(suggestions)

    kwargs: dict = {"seed": seed} if seed is not None else {}
    simulator = stormpy.simulator.create_simulator(program, **kwargs)
    simulator.set_action_mode(stormpy.simulator.SimulatorActionMode.GLOBAL_NAMES)

    # restart() returns the initial state as a JsonContainerDouble.
    initial_state = json.loads(str(simulator.restart()[0]))
    steps: list[dict] = [{"step": 0, "state": initial_state, "action": None}]

    for step_idx in range(nr_steps):
        if simulator.is_done():
            break
        actions = sorted(simulator.available_actions())
        if not actions:
            break
        action = actions[0]
        result = simulator.step(action)
        state = json.loads(str(result[0]))
        steps.append({"step": step_idx + 1, "state": state, "action": action})

    return {
        "path": steps,
        "final_state": steps[-1]["state"],
        "is_done": simulator.is_done(),
        "nr_steps_taken": len(steps) - 1,
    }


def parametric_check(
    prism_file_path: str,
    formula: str,
    constant_definitions: str = "",
) -> dict:
    """Perform parametric model checking and return the result as a rational function.

    Use this with models that contain uninstantiated probability parameters.
    """
    import stormpy.pars  # noqa: F401 — registers parametric model support

    path = resolve_prism_path(prism_file_path)
    program = stormpy.parse_prism_program(str(path))
    if constant_definitions.strip():
        program = _apply_constants(program, constant_definitions)

    properties = stormpy.parse_properties(formula, program)
    model = stormpy.build_parametric_model(program, properties)
    result = stormpy.model_checking(model, properties[0])

    initial_state = int(model.initial_states[0])
    result_func = result.at(initial_state)
    parameters = [str(p) for p in model.collect_all_parameters()]

    return {
        "result": str(result_func),
        "parameters": parameters,
        "model": _model_to_dict(model),
    }


def interval_check(
    prism_file_path: str,
    formula: str,
    constant_definitions: str = "",
    eps: float = 0.05,
) -> dict:
    """Build an interval model and check with both MAXIMIZE and MINIMIZE.

    Each transition probability *p* is widened to ``[max(0, p-eps), min(1, p+eps)]``.
    Returns both the min and max result for the initial state.
    """
    from stormpy.pycarl import Interval

    path = resolve_prism_path(prism_file_path)
    program = stormpy.parse_prism_program(str(path))
    program = _apply_constants(program, constant_definitions)

    properties = stormpy.parse_properties(formula, program)
    model = stormpy.build_model(program, properties)

    # Build interval transition matrix
    tm = model.transition_matrix
    is_mdp = model.model_type == stormpy.ModelType.MDP

    builder = stormpy.IntervalSparseMatrixBuilder(
        rows=0, columns=0, entries=0,
        force_dimensions=False,
        has_custom_row_grouping=is_mdp,
    )

    for state_id in range(model.nr_states):
        row_start = tm.get_row_group_start(state_id)
        row_end = tm.get_row_group_end(state_id)
        if is_mdp:
            builder.new_row_group(row_start)
        for row in range(row_start, row_end):
            for entry in tm.get_row(row):
                p = float(entry.value())
                lo = max(0.0, p - eps)
                hi = min(1.0, p + eps)
                builder.add_next_value(row, entry.column, Interval(lo, hi))

    interval_matrix = builder.build()
    components = stormpy.SparseIntervalModelComponents(transition_matrix=interval_matrix)
    components.state_labeling = model.labeling
    if len(model.reward_models) > 0:
        components.reward_models = model.reward_models

    if is_mdp:
        interval_model = stormpy.SparseIntervalMdp(components)
    else:
        interval_model = stormpy.SparseIntervalDtmc(components)

    # Check with MAXIMIZE
    env = stormpy.Environment()
    task_max = stormpy.CheckTask(properties[0].raw_formula, only_initial_states=False)
    task_max.set_uncertainty_resolution_mode(stormpy.UncertaintyResolutionMode.MAXIMIZE)
    if is_mdp:
        result_max = stormpy.check_interval_mdp(interval_model, task_max, env)
    else:
        result_max = stormpy.check_interval_dtmc(interval_model, task_max, env)

    # Check with MINIMIZE
    task_min = stormpy.CheckTask(properties[0].raw_formula, only_initial_states=False)
    task_min.set_uncertainty_resolution_mode(stormpy.UncertaintyResolutionMode.MINIMIZE)
    if is_mdp:
        result_min = stormpy.check_interval_mdp(interval_model, task_min, env)
    else:
        result_min = stormpy.check_interval_dtmc(interval_model, task_min, env)

    initial_state = int(interval_model.initial_states[0])

    return {
        "min_result": _to_python(result_min.at(initial_state)),
        "max_result": _to_python(result_max.at(initial_state)),
        "eps": eps,
        "model": _model_to_dict(interval_model),
    }


def get_transitions(
    prism_file_path: str,
    constant_definitions: str = "",
) -> dict:
    """Return the full sparse transition matrix as a list of dicts.

    Each entry has keys ``from``, ``action``, ``to``, and ``probability``.
    """
    path = resolve_prism_path(prism_file_path)
    program = stormpy.parse_prism_program(str(path))
    program = _apply_constants(program, constant_definitions)
    model = stormpy.build_model(program)

    transitions: list[dict] = []
    for state in model.states:
        for action in state.actions:
            for transition in action.transitions:
                transitions.append({
                    "from": int(state.id),
                    "action": int(action.id),
                    "to": int(transition.column),
                    "probability": _to_python(transition.value()),
                })

    return {
        "transitions": transitions,
        **_model_to_dict(model),
    }
