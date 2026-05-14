"""Policy-sampling transition updater for black-box policies.

Implements the methodology of "black-box policy verification via interval
DTMCs" (Wilson score interval + two-level Bonferroni correction + greedy
propagation through the MDP). Given an MDP M and a black-box policy pi,
constructs a SparseIntervalDtmc whose intervals contain the true induced
DTMC transition probabilities with global confidence 1 - alpha_joint.

Config string: ``policy_sampling;alpha=0.05;samples=500;half_width=0.05;seed=0``

At least one of ``samples`` / ``half_width`` must be given:
    - alpha + half_width  -> samples derived from N = ceil(z^2 / (4 h^2))
    - alpha + samples     -> achievable worst-case half-width reported
    - all three           -> honoured as given (warning on inconsistency)

The Bonferroni denominator is the exact total number of intervals
sum_s |Act(s)|, so non-uniform action availability is handled tightly.
"""
import json
import math
from typing import Optional

import numpy as np
import stormpy
from scipy.stats import norm
from stormpy.pycarl import Interval

from common.agents.black_box_agent import BlackBoxAgent
from common.transition_updaters.transition_updater import TransitionUpdater


class PolicySamplingTransitionUpdater(TransitionUpdater):

    def __init__(self, config_str: str):
        self.alpha: float = 0.05
        self.samples: Optional[int] = None
        self.half_width: Optional[float] = None
        self.seed: int = 0
        # Outer Bonferroni factor for multi-policy comparisons: alpha is
        # further divided by num_models so that K independently built IDTMCs
        # jointly hold with confidence >= 1 - alpha.
        self.num_models: int = 1
        super().__init__(config_str)

        if self.samples is None and self.half_width is None:
            raise ValueError(
                "policy_sampling updater requires either 'samples' or 'half_width'"
            )

    def parse_config(self, config_str: str) -> None:
        for part in config_str.split(";")[1:]:
            key, _, value = part.partition("=")
            key = key.strip()
            value = value.strip()
            if key == "":
                continue
            if key == "alpha":
                self.alpha = float(value)
            elif key == "samples":
                self.samples = int(value)
            elif key == "half_width":
                self.half_width = float(value)
            elif key == "seed":
                self.seed = int(value)
            elif key == "num_models":
                self.num_models = max(1, int(value))
            else:
                print(f"Warning: unknown policy_sampling parameter '{key}'")

    @property
    def requires_full_mdp(self) -> bool:
        return True

    def get_updater_name(self) -> str:
        return (f"policy_sampling(alpha={self.alpha},"
                f"samples={self.samples},half_width={self.half_width})")

    # ------------------------------------------------------------------ helpers

    # Name used in console output / MLflow param key prefix. Subclasses
    # (e.g. HoeffdingTransitionUpdater) override this.
    METHOD_NAME: str = "Wilson"
    METRIC_PREFIX: str = "ps"

    @staticmethod
    def _z_for(alpha_per: float) -> float:
        return float(norm.ppf(1.0 - alpha_per / 2.0))

    @staticmethod
    def _wilson_bounds(n_a: int, n: int, z: float) -> tuple:
        if n == 0:
            return 0.0, 1.0
        p_hat = n_a / n
        denom = 1.0 + z * z / n
        centre = (p_hat + z * z / (2.0 * n)) / denom
        half = (z * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n))
                ) / denom
        return max(0.0, centre - half), min(1.0, centre + half)

    # ---- CI hooks (overridden by Azuma-Hoeffding subclass) ----

    def _ci_radius(self, alpha_per: float, samples: int) -> float:
        """Worst-case half-width at this alpha_per and N (Wilson: p=1/2)."""
        z = self._z_for(alpha_per)
        return z / (2.0 * math.sqrt(samples))

    def _ci_samples_from_half_width(self, alpha_per: float,
                                    half_width: float) -> int:
        """Required N so that the worst-case half-width is <= half_width."""
        z = self._z_for(alpha_per)
        return int(math.ceil(z * z / (4.0 * half_width ** 2)))

    def _ci_bounds(self, n_a: int, samples: int, alpha_per: float) -> tuple:
        """Return (lo, hi) for one action's count under this CI."""
        z = self._z_for(alpha_per)
        return self._wilson_bounds(n_a, samples, z)

    def _banner_ci_lines(self, alpha_per: float, samples: int) -> list:
        """Extra lines to print in the banner describing the CI state."""
        z = self._z_for(alpha_per)
        return [f"  z-score                : {z:.4f}"]

    def _mlflow_ci_params(self, alpha_per: float, samples: int) -> dict:
        """Method-specific MLflow key/value pairs."""
        z = self._z_for(alpha_per)
        return {f"{self.METRIC_PREFIX}_z": z}

    @staticmethod
    def _greedy_bound(lo: np.ndarray, hi: np.ndarray, weights: np.ndarray,
                      maximize: bool) -> float:
        """Solve max/min sum_a pi_a * weights_a s.t. lo<=pi_a<=hi, sum pi_a=1."""
        order = np.argsort(-weights if maximize else weights)
        pi = lo.copy()
        budget = 1.0 - float(pi.sum())
        if budget < -1e-9:
            raise ValueError("Wilson lower bounds sum > 1; infeasible region")
        budget = max(0.0, budget)
        for a in order:
            if budget <= 0.0:
                break
            cap = float(hi[a] - lo[a])
            delta = min(budget, cap)
            pi[a] += delta
            budget -= delta
        return float(np.dot(pi, weights))

    def _resolve_sample_size(self, total_intervals: int) -> tuple:
        """Compute (samples, alpha_per, achievable_half_width)."""
        alpha_per = self.alpha / max(1, total_intervals * self.num_models)
        if self.samples is None:
            self.samples = max(
                self._ci_samples_from_half_width(alpha_per, self.half_width), 1)
        achievable_h = self._ci_radius(alpha_per, self.samples)
        if self.half_width is not None and achievable_h > self.half_width + 1e-9:
            print(f"Warning: requested samples={self.samples} only achieves "
                  f"worst-case half_width={achievable_h:.4g} > "
                  f"requested {self.half_width:.4g}")
        return self.samples, alpha_per, achievable_h

    @staticmethod
    def _log_to_mlflow(params: dict) -> None:
        try:
            import mlflow
            if mlflow.active_run() is None:
                return
            for k, v in params.items():
                mlflow.log_param(k, v)
        except Exception:
            pass

    # ------------------------------------------------------------------ main

    def update_model(self, model, env, agent):
        if not isinstance(agent, BlackBoxAgent):
            raise TypeError(
                "policy_sampling updater requires a BlackBoxAgent instance; "
                f"got {type(agent).__name__}"
            )
        if model.model_type != stormpy.ModelType.MDP:
            raise TypeError(
                "policy_sampling updater expects an MDP as input "
                f"(got {model.model_type})"
            )
        if not hasattr(model, "choice_labeling") or model.choice_labeling is None:
            raise RuntimeError(
                "policy_sampling updater requires the MDP to carry choice "
                "labels; ensure options.set_build_choice_labels(True)"
            )

        tm = model.transition_matrix
        choice_labeling = model.choice_labeling
        num_states = model.nr_states
        action_names_global = list(env.action_mapper.actions)

        # Pass 1: collect available actions per state to compute the Bonferroni
        # denominator sum_s |Act(s)|.
        available_per_state = []  # list[list[str]]
        for s in range(num_states):
            row_lo = tm.get_row_group_start(s)
            row_hi = tm.get_row_group_end(s)
            acts = []
            ord_counter = 0
            for row in range(row_lo, row_hi):
                labels = choice_labeling.get_labels_of_choice(row)
                if labels:
                    acts.append(next(iter(labels)))
                else:
                    # Unlabelled choice -> fall back to the action_mapper by
                    # ordinal position within the state's row group.
                    if ord_counter >= len(action_names_global):
                        raise RuntimeError(
                            f"State {s} has more choices than known actions; "
                            "cannot recover action name for unlabelled choice"
                        )
                    acts.append(action_names_global[ord_counter])
                ord_counter += 1
            available_per_state.append(acts)

        # Pre-compute per-action successor distributions so we can also detect
        # "policy-agnostic" states where every action at s has identical
        # Tr(s, a, .) — sampling at such states wastes queries.
        per_state_succs = []  # list[list[dict[succ_id -> prob]]]
        for s in range(num_states):
            row_lo = tm.get_row_group_start(s)
            row_hi = tm.get_row_group_end(s)
            rows = []
            for row in range(row_lo, row_hi):
                rows.append({e.column: float(e.value()) for e in tm.get_row(row)})
            per_state_succs.append(rows)

        # Storm-labelled deadlocks (states that were dead before the builder
        # inserted self-loops).
        deadlock_states = set()
        try:
            if model.labeling.contains_label("deadlock"):
                deadlock_states = set(model.labeling.get_states("deadlock"))
        except Exception:
            deadlock_states = set()

        def _is_skippable(s: int) -> str:
            """Return a short reason string if state s needs no sampling,
            else empty string."""
            if len(available_per_state[s]) == 1:
                return "single-action"
            if s in deadlock_states:
                return "deadlock"
            rows = per_state_succs[s]
            if rows and all(r == rows[0] for r in rows[1:]):
                return "policy-agnostic"
            return ""

        skippable_reasons = [_is_skippable(s) for s in range(num_states)]
        total_intervals = sum(len(a) for a in available_per_state)
        max_acts = max((len(a) for a in available_per_state), default=0)
        forced_states = sum(1 for r in skippable_reasons if r)
        sampled_states = num_states - forced_states

        samples, alpha_per, achievable_h = self._resolve_sample_size(
            total_intervals
        )

        print("=" * 64)
        print(f"Policy-sampling transition updater (CI: {self.METHOD_NAME})")
        print(f"  states                 : {num_states}")
        print(f"  sum |Act(s)|           : {total_intervals}")
        print(f"  max |Act(s)|           : {max_acts}")
        print(f"  num_models (K)         : {self.num_models}")
        print(f"  alpha_joint            : {self.alpha}")
        print(f"  alpha_per (Bonferroni) : {alpha_per:.6g}")
        for line in self._banner_ci_lines(alpha_per, samples):
            print(line)
        print(f"  samples per state (N)  : {samples}")
        print(f"  worst-case half-width  : {achievable_h:.4g}")
        # Break down skip reasons.
        by_reason = {}
        for r in skippable_reasons:
            if r:
                by_reason[r] = by_reason.get(r, 0) + 1
        print(f"  skippable states       : {forced_states} (sampling skipped)")
        for reason, count in sorted(by_reason.items()):
            print(f"    - {reason:<18} : {count}")
        print(f"  sampled states         : {sampled_states}")
        print(f"  total policy queries   : {samples * sampled_states}")
        print("=" * 64)

        mlflow_params = {
            f"{self.METRIC_PREFIX}_method": self.METHOD_NAME,
            f"{self.METRIC_PREFIX}_num_models": self.num_models,
            f"{self.METRIC_PREFIX}_alpha_joint": self.alpha,
            f"{self.METRIC_PREFIX}_alpha_per": alpha_per,
            f"{self.METRIC_PREFIX}_samples_per_state": samples,
            f"{self.METRIC_PREFIX}_num_states": num_states,
            f"{self.METRIC_PREFIX}_sum_act_s": total_intervals,
            f"{self.METRIC_PREFIX}_max_act_s": max_acts,
            f"{self.METRIC_PREFIX}_total_queries": samples * sampled_states,
            f"{self.METRIC_PREFIX}_forced_states": forced_states,
            f"{self.METRIC_PREFIX}_sampled_states": sampled_states,
            f"{self.METRIC_PREFIX}_worst_case_half_width": achievable_h,
        }
        mlflow_params.update(self._mlflow_ci_params(alpha_per, samples))
        self._log_to_mlflow(mlflow_params)

        rng = np.random.default_rng(self.seed)
        # Collected for a per-state debug summary printed after sampling.
        debug_rows = []

        # Build interval transition matrix (DTMC: one row per state).
        builder = stormpy.IntervalSparseMatrixBuilder(
            rows=0, columns=0, entries=0,
            force_dimensions=False,
            has_custom_row_grouping=False,
        )

        for s in range(num_states):
            avail_names = available_per_state[s]
            k = len(avail_names)

            # Map state id -> numpy feature vector.
            state_json = model.state_valuations.get_json(s)
            state_dict = json.loads(str(state_json))
            example = json.loads(env.storm_bridge.state_json_example)
            ordered = {key: state_dict[key] for key in example.keys()
                       if key in state_dict}
            state_vec = env.storm_bridge.parse_state(json.dumps(ordered))

            # Short-circuit states where the policy cannot change the
            # induced transition distribution:
            #   * |Act(s)| = 1 — no choice at all,
            #   * state labelled `deadlock` by Storm,
            #   * every available action has identical Tr(s, a, .) (absorbing
            #     with multiple actions, etc.) — any policy mix yields the
            #     same row.
            # In all three cases Wilson collapses to [1, 1] on (WLOG) the
            # first action; the black-box agent is not queried.
            avail_set = set(avail_names)
            counts = {name: 0 for name in avail_names}
            # Diagnostic breakdown — does NOT affect the Wilson math.
            # real_counts: agent produced an in-support action for this state.
            # unparsed_count: agent returned no parseable action (-1).
            # unavailable_count: agent returned an action not in Act(s).
            # Both unparsed and unavailable votes are remapped to the
            # alphabetically-first available action (cool_mc convention).
            real_counts = {name: 0 for name in avail_names}
            unparsed_count = 0
            unavailable_count = 0
            first_avail = sorted(avail_names)[0]
            skip_reason = skippable_reasons[s]
            if skip_reason:
                counts[avail_names[0]] = samples  # display only
                real_counts[avail_names[0]] = samples
                forced_single_action = True
            else:
                forced_single_action = False
                for _ in range(samples):
                    a_idx = int(agent.sample_action(
                        state_vec, available_actions=avail_names))
                    if 0 <= a_idx < len(action_names_global):
                        raw_name = action_names_global[a_idx]
                    else:
                        raw_name = None
                    if raw_name is None:
                        unparsed_count += 1
                        counts[first_avail] += 1
                    elif raw_name not in avail_set:
                        unavailable_count += 1
                        counts[first_avail] += 1
                    else:
                        real_counts[raw_name] += 1
                        counts[raw_name] += 1

            # Record for the debug summary.
            debug_rows.append({
                "state": s,
                "features": ordered,
                "counts": dict(counts),
                "real_counts": dict(real_counts),
                "unparsed": unparsed_count,
                "unavailable": unavailable_count,
                "available": list(avail_names),
                "forced": forced_single_action,
                "skip_reason": skip_reason,
            })

            # CI bounds per available action. For forced single-action
            # states we set pi(a|s) = [1, 1] directly.
            lo = np.zeros(k)
            hi = np.zeros(k)
            if forced_single_action:
                lo[0] = 1.0
                hi[0] = 1.0
            else:
                for i, name in enumerate(avail_names):
                    lo[i], hi[i] = self._ci_bounds(
                        counts[name], samples, alpha_per)

            # Project lower bounds down so they are jointly feasible
            # (sum lo <= 1). Wilson alone can violate this for small N.
            slo = float(lo.sum())
            if slo > 1.0:
                lo = lo * (1.0 / slo)
            # Cap upper bounds so each is reachable given others at lo.
            hi = np.minimum(hi, 1.0 - (lo.sum() - lo))

            # Per (s, a) successor distribution Tr(s, a, .) — already cached
            # in per_state_succs[s] by the pre-pass above.
            per_action_succ = per_state_succs[s]

            # Aggregate successor set.
            succ_ids = sorted({sid for d in per_action_succ for sid in d})

            # For each successor s', greedy on weights w_a = Tr(s, a, s').
            # Storm's interval DTMC reachability requires graph-preservation:
            # every edge with upper > 0 must also have lower > 0 so the graph
            # topology is invariant under adversarial interval resolution.
            # We clamp with a tiny epsilon well below our sampling noise.
            graph_eps = 1e-12
            for sp in succ_ids:
                w = np.array([d.get(sp, 0.0) for d in per_action_succ])
                lower = self._greedy_bound(lo, hi, w, maximize=False)
                upper = self._greedy_bound(lo, hi, w, maximize=True)
                lower = max(0.0, min(1.0, lower))
                upper = max(lower, min(1.0, upper))
                if upper > 0.0:
                    lower = max(lower, graph_eps)
                    upper = max(upper, lower)
                builder.add_next_value(s, sp, Interval(lower, upper))

        interval_matrix = builder.build()
        components = stormpy.SparseIntervalModelComponents(
            transition_matrix=interval_matrix
        )
        components.state_labeling = model.labeling
        # Reward models on MDP choices cannot be reused on the DTMC directly;
        # skip for now (verification of reward properties on the IDTMC would
        # require marginalising rewards over the empirical policy too).

        interval_dtmc = stormpy.SparseIntervalDtmc(components)
        print(f"Built IDTMC: {interval_dtmc.nr_states} states, "
              f"{interval_dtmc.nr_transitions} transitions")

        # Per-state sampling summary (useful for debugging LLM policies on
        # small MDPs).
        if debug_rows:
            all_actions = sorted(
                {a for row in debug_rows for a in row["available"]}
            )
            header = "state | features".ljust(44) + " | " + \
                     " ".join(f"{a:>10}" for a in all_actions) + \
                     " | unparsed  unavail"
            print("Per-state sampling counts (real votes; "
                  "unparsed/unavail remapped to alphabetical-first fallback):")
            print(header)
            print("-" * len(header))
            total_unparsed = 0
            total_unavailable = 0
            for row in debug_rows:
                feats = ",".join(f"{k}={v}" for k, v in row["features"].items())
                # Show REAL votes in the per-action columns so the column for
                # the fallback action isn't inflated by parse failures.
                src = row.get("real_counts", row["counts"])
                cols = " ".join(
                    f"{src[a]:>10}" if a in src
                    else f"{'-':>10}"
                    for a in all_actions
                )
                up = row.get("unparsed", 0)
                un = row.get("unavailable", 0)
                total_unparsed += up
                total_unavailable += un
                tag = f" ({row['skip_reason']}, no sampling)" if row.get("forced") else ""
                print(f"{row['state']:<5} | {feats:<38} | {cols} | "
                      f"{up:>8}  {un:>7}{tag}")
            if total_unparsed or total_unavailable:
                print(f"TOTAL across sampled states: "
                      f"unparsed={total_unparsed}, "
                      f"unavailable={total_unavailable} "
                      f"(both remapped to alphabetical-first action)")

        return interval_dtmc
