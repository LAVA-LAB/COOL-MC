"""Multi-policy IDTMC comparison driver.

Spawns cool_mc.py K times (once per policy) with a shared
``num_models=K`` Bonferroni split, loads the resulting IDTMCs from a
temporary directory, and prints a side-by-side verification table +
pairwise DISJOINT/OVERLAP verdicts + per-transition overlap coefficients.
The .drn files are discarded when the comparison finishes.

Example:
    python compare_idtmcs.py \
        --prism_file_path frozen_lake.prism \
        --constant_definitions "control=0.8,start_position=0" \
        --prop 'P=? [ F "at_frisbee" ]' \
        --alpha 0.1 --samples 10 --seed 128 \
        --algorithms "ollama_agent;model=gemma3:1b;temperature=0.2;prompt_file=../prism_files/frozen_lake_prompt.txt;host=http://host.docker.internal:11434#ollama_agent;model=gemma3:1b;temperature=1.0;prompt_file=../prism_files/frozen_lake_prompt.txt;host=http://host.docker.internal:11434" \
        --labels "cold,hot"

Policy configs are separated by '#'. Passing a single policy is allowed
and reduces to a regular single-policy run with a tight num_models=1
split.
"""
import argparse
import os
import subprocess
import sys
import tempfile

import stormpy


COOL_MC_PY = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                          "cool_mc.py"))


def _parse_algo_config(algo: str) -> dict:
    out = {"_name": algo.split(";", 1)[0].strip()}
    for part in algo.split(";")[1:]:
        k, _, v = part.partition("=")
        out[k.strip()] = v.strip()
    return out


def unload_ollama_model(algo: str) -> None:
    """Force Ollama to evict the policy's model from memory.

    Called between sequential policy runs so the container / host does
    not hold multiple big LLMs resident at once.
    """
    cfg = _parse_algo_config(algo)
    if cfg.get("_name") != "ollama_agent":
        return
    host = cfg.get("host", "http://host.docker.internal:11434").rstrip("/")
    model = cfg.get("model")
    if not model:
        return
    try:
        import requests
        requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": "", "stream": False,
                  "keep_alive": 0},
            timeout=30,
        )
        print(f"[compare] requested unload of {model} from {host}")
    except Exception as exc:
        print(f"[compare] unload request failed for {model}: {exc}")


def run_cool_mc(algorithm: str, idtmc_out: str, *, alpha: float, samples: int,
                num_models: int, prism_dir: str, prism_file_path: str,
                constant_definitions: str, prop: str, seed: int,
                project_name: str, bound: str) -> None:
    tu = (f"{bound};alpha={alpha};samples={samples};"
          f"num_models={num_models};seed={seed}")
    cmd = [
        sys.executable, COOL_MC_PY,
        "--task=rl_model_checking",
        f"--project_name={project_name}",
        f"--algorithm={algorithm}",
        f"--prism_dir={prism_dir}",
        f"--prism_file_path={prism_file_path}",
        f"--constant_definitions={constant_definitions}",
        f"--prop={prop}",
        f"--transition_updater={tu}",
        f"--idtmc_out={idtmc_out}",
        f"--seed={seed}",
    ]
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    print("\n$ (cwd=" + cwd + ") " + " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def load_idtmc(path: str):
    if not os.path.exists(path):
        sys.exit(f"IDTMC file not found: {path}")
    return stormpy.build_interval_model_from_drn(path)


def check_bounds(model, formula_str: str):
    props = stormpy.parse_properties_without_context(formula_str)
    env = stormpy.Environment()
    s0 = model.initial_states[0]
    task_max = stormpy.CheckTask(props[0].raw_formula, only_initial_states=False)
    task_max.set_uncertainty_resolution_mode(
        stormpy.UncertaintyResolutionMode.MAXIMIZE)
    task_min = stormpy.CheckTask(props[0].raw_formula, only_initial_states=False)
    task_min.set_uncertainty_resolution_mode(
        stormpy.UncertaintyResolutionMode.MINIMIZE)
    if model.model_type == stormpy.ModelType.DTMC:
        res_max = stormpy.check_interval_dtmc(model, task_max, env).at(s0)
        res_min = stormpy.check_interval_dtmc(model, task_min, env).at(s0)
    else:
        res_max = stormpy.check_interval_mdp(model, task_max, env).at(s0)
        res_min = stormpy.check_interval_mdp(model, task_min, env).at(s0)
    lo = min(float(res_min), float(res_max))
    hi = max(float(res_min), float(res_max))
    return lo, hi


def collect_interval_transitions(model):
    tm = model.transition_matrix
    out = {}
    for s in range(model.nr_states):
        for entry in tm.get_row(s):
            iv = entry.value()
            out[(s, entry.column)] = (float(iv.lower()), float(iv.upper()))
    return out


def overlap_coefficient(a, b):
    lo_a, hi_a = a
    lo_b, hi_b = b
    inter = max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))
    min_width = min(hi_a - lo_a, hi_b - lo_b)
    if min_width <= 0.0:
        return 1.0 if (lo_a == lo_b and hi_a == hi_b) else 0.0
    return inter / min_width


def pairwise_overlap_avg(models):
    scores = {}
    transitions = [collect_interval_transitions(m) for m in models]
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            shared = set(transitions[i]) & set(transitions[j])
            if not shared:
                scores[(i, j)] = 0.0
                continue
            scores[(i, j)] = sum(
                overlap_coefficient(transitions[i][k], transitions[j][k])
                for k in shared) / len(shared)
    return scores


def intervals_disjoint(a, b):
    lo_a, hi_a = a
    lo_b, hi_b = b
    return hi_a < lo_b or hi_b < lo_a


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algorithms", required=True,
                    help="'#'-separated policy configs. One cool_mc run per policy.")
    ap.add_argument("--labels", default="",
                    help="Comma-separated display names (defaults to policy_0...).")
    ap.add_argument("--prop", required=True)
    ap.add_argument("--prism_dir", default="../prism_files")
    ap.add_argument("--prism_file_path", required=True)
    ap.add_argument("--constant_definitions", default="")
    ap.add_argument("--alpha", type=float, default=None,
                    help="Joint failure probability (e.g. 0.05). Mutually exclusive with --confidence.")
    ap.add_argument("--confidence", type=float, default=None,
                    help="Joint confidence level in [0,1] (e.g. 0.95) or as a percentage (e.g. 95). "
                         "Equivalent to --alpha=(1-confidence). Mutually exclusive with --alpha.")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--seed", type=int, default=128)
    ap.add_argument("--project_name", default="idtmc_comparison")
    ap.add_argument("--bound", default="policy_sampling",
                    choices=["policy_sampling", "hoeffding",
                             "empirical_bernstein"],
                    help="Concentration bound used to wrap per-action "
                         "empirical frequencies in confidence intervals.")
    return ap.parse_args()


def resolve_alpha(args) -> float:
    if args.alpha is not None and args.confidence is not None:
        sys.exit("Pass either --alpha or --confidence, not both.")
    if args.confidence is not None:
        c = args.confidence
        if c > 1.0:  # accept percentage form, e.g. 95 -> 0.95
            c /= 100.0
        if not 0.0 < c < 1.0:
            sys.exit(f"--confidence must lie in (0,1) or (0,100); got {args.confidence}")
        return 1.0 - c
    return args.alpha if args.alpha is not None else 0.05


def main():
    args = parse_args()
    args.alpha = resolve_alpha(args)
    algorithms = [a for a in args.algorithms.split("#") if a]
    K = len(algorithms)
    if K == 0:
        sys.exit("--algorithms must contain at least one config")

    labels = [s.strip() for s in args.labels.split(",") if s.strip()]
    if len(labels) != K:
        labels = [f"policy_{i}" for i in range(K)]

    print("=" * 72)
    print(f"Multi-policy IDTMC comparison (K = {K})")
    print(f"  property         : {args.prop}")
    print(f"  joint confidence : {1.0 - args.alpha:.4f}")
    print(f"  samples / policy : {args.samples}")
    print(f"  bound            : {args.bound}")
    print("=" * 72)

    with tempfile.TemporaryDirectory(prefix="idtmc_compare_") as tmp:
        paths = []
        for lbl, algo in zip(labels, algorithms):
            out = os.path.join(tmp, f"{lbl}.drn")
            run_cool_mc(
                algorithm=algo,
                idtmc_out=out,
                alpha=args.alpha,
                samples=args.samples,
                num_models=K,
                prism_dir=args.prism_dir,
                prism_file_path=args.prism_file_path,
                constant_definitions=args.constant_definitions,
                prop=args.prop,
                seed=args.seed,
                project_name=f"{args.project_name}_{lbl}",
                bound=args.bound,
            )
            paths.append(out)
            # Free this policy's LLM before the next one loads so VRAM /
            # RAM can only hold one model at a time.
            unload_ollama_model(algo)

        models = [load_idtmc(p) for p in paths]
        bounds = [check_bounds(m, args.prop) for m in models]

        print("\n" + "=" * 72)
        print("Per-policy verification bounds (adversarial semantics)")
        print("=" * 72)
        print(f"{'policy':<20} {'lower':>12} {'upper':>12}   "
              f"{'states':>7} {'transitions':>12}")
        for lbl, m, (lo, hi) in zip(labels, models, bounds):
            print(f"{lbl:<20} {lo:>12.6f} {hi:>12.6f}   "
                  f"{m.nr_states:>7} {m.nr_transitions:>12}")

        print("\nPairwise verdicts:")
        any_overlap = False
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                disjoint = intervals_disjoint(bounds[i], bounds[j])
                verdict = ("DISJOINT (strict ordering)"
                           if disjoint
                           else "OVERLAP (indistinguishable at this N)")
                any_overlap = any_overlap or (not disjoint)
                print(f"  {labels[i]} vs {labels[j]}: {verdict}")

        print("\nStructural interval overlap (paper coefficient):")
        scores = pairwise_overlap_avg(models)
        for (i, j), s in scores.items():
            print(f"  {labels[i]} vs {labels[j]}: mean overlap = {s:.4f}")

        print(f"\nGuarantee: all {K} interval models jointly contain their")
        print(f"true induced DTMCs with probability >= {1.0 - args.alpha:.4f}.")

        try:
            import mlflow
            if mlflow.active_run() is None:
                mlflow.start_run(run_name=args.project_name)
            for lbl, (lo, hi) in zip(labels, bounds):
                mlflow.log_metric(f"prob_lower_{lbl}", lo)
                mlflow.log_metric(f"prob_upper_{lbl}", hi)
            for (i, j), s in scores.items():
                mlflow.log_metric(
                    f"overlap_{labels[i]}_vs_{labels[j]}", s)
            mlflow.log_metric("any_overlap", int(any_overlap))
            mlflow.log_param("num_models", K)
            mlflow.log_param("alpha_joint", args.alpha)
        except Exception as exc:
            print(f"(mlflow logging skipped: {exc})")


if __name__ == "__main__":
    main()
