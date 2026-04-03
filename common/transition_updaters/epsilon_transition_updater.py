"""Epsilon transition updater.

Converts a SparseDtmc / SparseMdp into a SparseIntervalDtmc / SparseIntervalMdp
by replacing each transition probability p with the interval
[max(0, p - eps), min(1, p + eps)].
"""
import stormpy
from stormpy.pycarl import Interval
from common.transition_updaters.transition_updater import TransitionUpdater


class EpsilonTransitionUpdater(TransitionUpdater):

    def __init__(self, config_str: str):
        self.eps = 0.05  # default
        super().__init__(config_str)

    def parse_config(self, config_str: str) -> None:
        for part in config_str.split(";")[1:]:
            key, _, value = part.partition("=")
            if key.strip() == "eps":
                self.eps = float(value.strip())

    def get_updater_name(self) -> str:
        return f"epsilon(eps={self.eps})"

    def update_model(self, model, env, agent):
        print(f"Applying epsilon transition updater (eps={self.eps}) ...")

        tm = model.transition_matrix
        num_states = model.nr_states
        is_mdp = model.model_type == stormpy.ModelType.MDP

        builder = stormpy.IntervalSparseMatrixBuilder(
            rows=0, columns=0, entries=0,
            force_dimensions=False,
            has_custom_row_grouping=is_mdp,
        )

        for state_id in range(num_states):
            row_start = tm.get_row_group_start(state_id)
            row_end = tm.get_row_group_end(state_id)

            if is_mdp:
                builder.new_row_group(row_start)

            for row in range(row_start, row_end):
                for entry in tm.get_row(row):
                    p = float(entry.value())
                    lo = max(0.0, p - self.eps)
                    hi = min(1.0, p + self.eps)
                    builder.add_next_value(row, entry.column, Interval(lo, hi))

        interval_matrix = builder.build()

        components = stormpy.SparseIntervalModelComponents(
            transition_matrix=interval_matrix
        )
        components.state_labeling = model.labeling

        if len(model.reward_models) > 0:
            components.reward_models = model.reward_models

        if is_mdp:
            interval_model = stormpy.SparseIntervalMdp(components)
        else:
            interval_model = stormpy.SparseIntervalDtmc(components)

        print(f"Interval model: {interval_model.nr_states} states, "
              f"{interval_model.nr_transitions} transitions")

        return interval_model
