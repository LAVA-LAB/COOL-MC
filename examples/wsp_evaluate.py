"""Evaluation of the weight-space programming (WSP) experiments. Makes NO LLM calls.

Called by examples/weight_space_programming_experiments.sh:

  optimum --prism FILE --constants C --prop P
      Storm on the MDP; prints "OPTIMUM <optimal probability> <number of MDP states>".

  report --out DIR --tol T --three-jobs-constants C --three-jobs-prop P
      Model checks the networks that every run exported (after Step 1, after Step 2, and the
      best verified network), computes their accuracy on the all-optimal-actions dataset,
      and prints the tables of the paper (environments, RQ1-RQ4).

A run directory DIR/runs/<id>/ contains run.json (written by the bash script), the agent's
exports (wsp_step1.json, wsp_step2.json, wsp_best.json, wsp_mc.json, wsp_calls.json) and one
status file: DONE, CREATION_FAILED (the LLM wrote no valid network at all) or FAILED.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

MODEL_NAMES = {"deepseek-v4-pro:cloud": "DeepSeek-V4-Pro", "kimi-k3.1-pro:cloud": "Kimi-K3.1-Pro"}
FINAL = ("done", "creation_failed")   # runs that count in the tables


class Quiet:
    """Silence stdout/stderr on file-descriptor level (Storm and cool-mc print a lot)."""

    def __enter__(self):
        sys.stdout.flush()
        sys.stderr.flush()
        self.saved = (os.dup(1), os.dup(2))
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)

    def __exit__(self, *exc):
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(self.saved[0], 1)
        os.dup2(self.saved[1], 2)
        os.close(self.saved[0])
        os.close(self.saved[1])
        return False


# --------------------------------------------------------------------------- #
# Storm / cool-mc helpers
# --------------------------------------------------------------------------- #
def mdp_optimum(prism, constants, prop):
    """Optimal probability of `prop` on the MDP and the number of MDP states."""
    import stormpy
    with Quiet():
        prog = stormpy.parse_prism_program(prism)
        prog = stormpy.preprocess_symbolic_input(prog, [], constants)[0].as_prism_program()
        props = stormpy.parse_properties_for_prism_program(prop, prog)
        model = stormpy.build_model(prog, props)
        result = stormpy.model_checking(model, props[0])
        return float(result.at(model.initial_states[0])), int(model.nr_states)


_ENVS, _DATASETS, _OPTIMA = {}, {}, {}


def get_env(prism, constants, reward_flag):
    key = (prism, constants, bool(reward_flag))
    if key not in _ENVS:
        from common.safe_gym.safe_gym import SafeGym
        with Quiet():
            _ENVS[key] = SafeGym(prism, constants, 100, 1000, bool(reward_flag), 0, "")
    return _ENVS[key]


def get_optimum(prism, constants, prop):
    key = (prism, constants, prop)
    if key not in _OPTIMA:
        _OPTIMA[key] = mdp_optimum(prism, constants, prop)
    return _OPTIMA[key]


def get_dataset(prism, constants, prop, env):
    """All optimal state-action pairs, built exactly as in Step 2 of the pipeline."""
    key = (prism, constants, prop)
    if key not in _DATASETS:
        from common.behavioral_cloning_dataset.all_optimal_dataset import AllOptimalDataset
        with Quiet():
            dataset = AllOptimalDataset(f"all_optimal_dataset;{prism};{prop};{constants}")
            dataset.create(env)
            data = dataset.get_data()
        _DATASETS[key] = (data["X_train"], data["y_train"])
    return _DATASETS[key]


def make_agent(env, run, prop):
    from common.agents.weight_space_programming_agent import WeightSpaceProgrammingAgent
    with Quiet():
        return WeightSpaceProgrammingAgent("weight_space_programming_agent;model=none;debug=0",
                                           env.observation_space.shape[0],
                                           [run["neurons"]] * run["layers"],
                                           len(env.action_mapper.actions), env.action_mapper.actions,
                                           prop=prop)


def load_layers(path):
    with open(path) as f:
        data = json.load(f)
    return [(np.asarray(l["W"], dtype=np.float32), np.asarray(l["b"], dtype=np.float32))
            for l in data["layers"]]


def query(env, agent, constants, prop):
    """Probability of `prop` in the DTMC induced by the agent's network (COOL-MC + Storm)."""
    with Quiet():
        result, _ = env.storm_bridge.model_checker.induced_markov_chain(agent, None, env, constants, prop)
    return float(result)


def is_optimal(q, optimum, prop, tol):
    if q is None:
        return False
    return q <= optimum + tol if "min" in prop.split("=")[0] else q >= optimum - tol


def read_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def load_runs(out):
    runs = []
    for d in sorted(glob.glob(os.path.join(out, "runs", "*"))):
        run = read_json(os.path.join(d, "run.json"))
        if run is None:
            continue
        run["dir"] = d
        run["id"] = os.path.basename(d)
        for marker, status in (("DONE", "done"), ("CREATION_FAILED", "creation_failed"), ("FAILED", "failed")):
            if os.path.exists(os.path.join(d, marker)):
                run["status"] = status
                with open(os.path.join(d, marker)) as f:
                    text = f.read().strip()
                run["seconds"] = int(text) if text.isdigit() else None
                break
        else:
            run["status"], run["seconds"] = "incomplete", None
        runs.append(run)
    runs.sort(key=lambda r: (r["rq"], r["order"], r["model"], r["seed"]))
    return runs


def evaluate_run(run, tol, warnings, with_accuracy):
    """Query result (and accuracy) of the networks after Step 1, Step 2 and Step 3."""
    env = get_env(run["prism"], run["constants"], run["reward_flag"])
    agent = make_agent(env, run, run["prop"])
    X = y = None
    if with_accuracy:
        X, y = get_dataset(run["prism"], run["constants"], run["prop"], env)
    files = {"step1": "wsp_step1.json", "step2": "wsp_step2.json", "best": "wsp_best.json"}
    for stage, name in files.items():
        path = os.path.join(run["dir"], name)
        if stage == "step2" and not os.path.exists(path):
            path = os.path.join(run["dir"], files["step1"])   # Step 2 had no dataset to work on
        if not os.path.exists(path):
            run[stage] = None
            continue
        layers = load_layers(path)
        agent.set_weights(layers)
        q = query(env, agent, run["constants"], run["prop"])
        acc = agent._accuracy(X, y)[0] if with_accuracy and X is not None and len(X) else None
        run[stage] = {"q": q, "acc": acc, "opt": is_optimal(q, run["optimum"], run["prop"], tol),
                      "layers": layers}

    run["mc"] = (read_json(os.path.join(run["dir"], "wsp_mc.json"), {}) or {}).get("results", [])
    run["calls"] = read_json(os.path.join(run["dir"], "wsp_calls.json"), []) or []
    run["requests"] = sum(1 for c in run["calls"] if c["outcome"] != "fallback_accepted")

    # Consistency checks between the pipeline's own model checking and this re-evaluation.
    if run["mc"] and run["step2"] and abs(run["step2"]["q"] - run["mc"][0]) > 1e-6:
        warnings.append(f"{run['id']}: network after Step 2 gives {run['step2']['q']:.6g}, but the "
                        f"first model-checking round of the run reported {run['mc'][0]:.6g}")
    if run["mc"] and run["best"]:
        expected = min(run["mc"]) if "min" in run["prop"].split("=")[0] else max(run["mc"])
        if abs(run["best"]["q"] - expected) > 1e-6:
            warnings.append(f"{run['id']}: best network gives {run['best']['q']:.6g}, "
                            f"but the best model-checking result of the run was {expected:.6g}")
    if run["status"] == "done" and run["best"] is None:
        warnings.append(f"{run['id']}: finished without a verified network (wsp_best.json missing)")


def check_network(run, stage, constants, prop):
    """Model check the network of `run` (after `stage`) on another model/property."""
    if not run.get(stage):
        return None
    env = get_env(run["prism"], constants, run["reward_flag"])
    agent = make_agent(env, run, prop)
    agent.set_weights(run[stage]["layers"])
    return query(env, agent, constants, prop)


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #
def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def fmt(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def print_table(title, headers, rows, note=None):
    widths = [max(len(str(x)) for x in col) for col in zip(headers, *rows)] if rows else [len(h) for h in headers]
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    print(f"\n{title}")
    print(line)
    print("| " + " | ".join(str(h).ljust(w) for h, w in zip(headers, widths)) + " |")
    print(line)
    for row in rows:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(row, widths)) + " |")
    print(line)
    if note:
        print(note)


def group(runs, key):
    groups = {}
    for run in runs:
        groups.setdefault(key(run), []).append(run)
    return groups


TABLE1_LABELS = {"transporter": "Transporter (2 jobs)"}
LATEX_QUERIES = {
    "Pmax=? [ F pos=15 ]": r"$P(\Diamond\, \mathit{goal})$",
    "Pmax=? [ F jobs_done=2 ]": r"$P(\Diamond\, \mathit{jobs\_done} = 2)$",
    "Pmax=? [ F jobs_done=3 ]": r"$P(\Diamond\, \mathit{jobs\_done} = 3)$",
    "Pmax=? [ !(passenger & fuel<4) U jobs_done=2 ]":
        r"$P(\neg(\mathit{passenger} \wedge \mathit{fuel} < 4)\ U\ \mathit{jobs\_done} = 2)$",
    'Pmax=? [ F "success" ]': r"$P(\Diamond\, \mathit{success})$",
}


def tex(value, digits):
    return "--" if value is None else f"{value:.{digits}f}"


def tex_probability(value):
    if value is None:
        return "--"
    return f"${int(round(value))}$" if abs(value - round(value)) < 1e-9 else f"${value:.3f}$"


def tex_query(prop):
    if prop in LATEX_QUERIES:
        return LATEX_QUERIES[prop]
    return r"\texttt{" + prop.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&") + "}"


def stage_stats(rs, stage):
    values = [r.get(stage) or {} for r in rs]
    return {"acc": mean([v.get("acc") for v in values]), "q": mean([v.get("q") for v in values]),
            "opt": sum(bool(v.get("opt")) for v in values), "n": len(rs)}


def report(args):
    runs = load_runs(args.out)
    print("=" * 78)
    print("Weight-space programming experiments: results (all numbers computed by Storm)")
    print("=" * 78)
    if not runs:
        print(f"No runs found in {args.out}/runs.")
        return 1

    counts = {s: sum(r["status"] == s for r in runs) for s in ("done", "creation_failed", "failed", "incomplete")}
    print(f"Runs found: {len(runs)} | finished: {counts['done']} | LLM wrote no valid network: "
          f"{counts['creation_failed']} | failed: {counts['failed']} | incomplete: {counts['incomplete']}")
    for r in runs:
        if r["status"] in ("failed", "incomplete"):
            print(f"  not counted: {r['id']} ({r['status']}), log: {r['dir']}/run.log")

    final = [r for r in runs if r["status"] in FINAL]
    rq1 = [r for r in final if r["rq"] == "rq1"]
    rq3 = [r for r in final if r["rq"] == "rq3"]
    warnings = []
    for i, r in enumerate(final, 1):
        print(f"  evaluating {i}/{len(final)}: {r['id']}", flush=True)
        evaluate_run(r, args.tol, warnings, with_accuracy=(r["rq"] == "rq1"))
    name = lambda model: MODEL_NAMES.get(model, model)  # noqa: E731

    # ---- collect all numbers once; text and LaTeX are rendered from them ----------------
    env_rows, seen = [], set()
    for r in sorted(runs, key=lambda r: (r["rq"], r["order"])):
        key = (r["prism"], r["constants"], r["prop"])
        if key not in seen:
            seen.add(key)
            env_rows.append({"label": r["label"], "latex_label": TABLE1_LABELS.get(r["env"], r["label"]), "spec": r})
    transporter = next((r for r in runs if r["rq"] == "rq1" and r["env"] == "transporter"), None)
    if transporter is not None:
        three = dict(transporter, constants=args.three_jobs_constants, prop=args.three_jobs_prop)
        pos = next(i for i, row in enumerate(env_rows) if row["spec"]["env"] == "transporter") + 1
        env_rows.insert(pos, {"label": "Transporter (3 jobs)", "latex_label": "Transporter (3 jobs)", "spec": three})
    for row in env_rows:
        s = row["spec"]
        env = get_env(s["prism"], s["constants"], s["reward_flag"])
        row["d"], row["actions"] = env.observation_space.shape[0], len(env.action_mapper.actions)
        row["optimum"], row["states"] = get_optimum(s["prism"], s["constants"], s["prop"])

    rq1_rows = []
    for (_, label, model), rs in group(rq1, lambda r: (r["order"], r["label"], r["model"])).items():
        rq1_rows.append({"label": label, "model": model, "calls": mean([r["requests"] for r in rs]),
                         **{stage: stage_stats(rs, stage) for stage in ("step1", "step2", "best")}})
    rq1_all = {model: {"calls": mean([r["requests"] for r in rs]),
                       **{stage: stage_stats(rs, stage) for stage in ("step1", "step2", "best")}}
               for model, rs in group(rq1, lambda r: r["model"]).items()}

    rq2_rows = {}
    if transporter is not None:
        opt3, _ = get_optimum(transporter["prism"], args.three_jobs_constants, args.three_jobs_prop)
        for model, rs in group([r for r in rq1 if r["env"] == "transporter"], lambda r: r["model"]).items():
            q3 = [check_network(r, "best", args.three_jobs_constants, args.three_jobs_prop) for r in rs]
            rq2_rows[model] = {"n": len(rs), "opt2": sum(bool((r.get("best") or {}).get("opt")) for r in rs),
                               "opt3": sum(is_optimal(q, opt3, args.three_jobs_prop, args.tol) for q in q3),
                               "q3": mean(q3)}

    rq3_rows, safety = {}, (rq3[0] if rq3 else None)
    for model, rs in group(rq3, lambda r: r["model"]).items():
        nets = [r for r in rq1 if r["env"] == "transporter" and r["model"] == model]
        sat = [is_optimal(check_network(r, "best", safety["constants"], safety["prop"]),
                          safety["optimum"], safety["prop"], args.tol) for r in nets]
        rq3_rows[model] = {"rq1_sat": sum(sat), "rq1_n": len(nets),
                           "opt": sum(bool((r.get("best") or {}).get("opt")) for r in rs), "n": len(rs),
                           "q": mean([(r.get("best") or {}).get("q") for r in rs]),
                           "rounds": mean([len(r["mc"]) for r in rs]), "calls": mean([r["requests"] for r in rs])}

    rq4_rows = {}
    for model, rs in group(rq1 + rq3, lambda r: r["model"]).items():
        outcomes = [c["outcome"] for r in rs for c in r["calls"]]
        n = {o: outcomes.count(o) for o in set(outcomes)}
        rq4_rows[model] = {
            "answers": sum(v for o, v in n.items() if o not in ("request_failed", "fallback_accepted")),
            "invalid_json": n.get("invalid_json", 0), "wrong_shape": n.get("wrong_shape", 0),
            "self_check": n.get("self_check_failed", 0) + n.get("accepted_despite_failure", 0),
            "accepted": n.get("accepted_despite_failure", 0) + n.get("fallback_accepted", 0),
            "failed_requests": n.get("request_failed", 0)}

    # ---- text tables -------------------------------------------------------------------
    print_table("Table 1: Environments (optimum = best probability of any policy, Storm on the MDP)",
                ["Environment", "d", "|Act|", "|S|", "Network", "PCTL query", "Optimum"],
                [[row["label"], row["d"], row["actions"], f"{row['states']:,}",
                  f"{row['spec']['layers']} x {row['spec']['neurons']}", row["spec"]["prop"],
                  fmt(row["optimum"], 6)] for row in env_rows])

    rows = []
    for r in final:
        stage = {s: r.get(s) or {} for s in ("step1", "step2", "best")}
        rows.append([r["id"], "ok" if r["status"] == "done" else "no valid network",
                     fmt(stage["step1"].get("acc"), 2), fmt(stage["step1"].get("q")),
                     fmt(stage["step2"].get("acc"), 2), fmt(stage["step2"].get("q")),
                     fmt(stage["best"].get("acc"), 2), fmt(stage["best"].get("q")),
                     "yes" if stage["best"].get("opt") else "no", len(r["mc"]), r["requests"],
                     "-" if r["seconds"] is None else f"{r['seconds'] / 60:.1f}"])
    print_table("All runs (Acc. in %, q = query result; S1/S2/S3 = after Step 1/2/3)",
                ["Run", "Status", "Acc S1", "q S1", "Acc S2", "q S2", "Acc S3", "q* S3", "Optimal",
                 "MC rounds", "LLM calls", "Minutes"], rows)

    def rq1_text(label, model, row):
        cells = [label, name(model)]
        for stage in ("step1", "step2", "best"):
            st = row[stage]
            cells += [fmt(st["acc"], 2), fmt(st["q"]), f"{st['opt']}/{st['n']}"]
        return cells + [fmt(row["calls"], 1)]

    print_table("Table 2 (RQ1): accuracy and query results after each step",
                ["Environment", "LLM", "Acc S1", "q S1", "Opt S1", "Acc S2", "q S2", "Opt S2",
                 "Acc S3", "q* S3", "Opt S3", "Calls"],
                [rq1_text(row["label"], row["model"], row) for row in rq1_rows]
                + [rq1_text("All environments", model, row) for model, row in rq1_all.items()],
                "Acc: mean accuracy (%) on the all-optimal-actions dataset; q: mean query result; "
                f"Opt: runs that reach the optimum\n(tolerance {args.tol:g}); Calls: mean LLM requests per run. "
                "'All environments' gives the numbers for the introduction;\nOpt S2 = optimal runs "
                "without model-checking feedback.")

    print_table(f"Table 3 (RQ2): transporter networks for 2 jobs, model checked with 3 jobs "
                f"({args.three_jobs_prop}, {args.three_jobs_constants})",
                ["LLM", "Optimal for 2 jobs", "Optimal for 3 jobs", "Mean P(F jobs_done=3)"],
                [[name(m), f"{v['opt2']}/{v['n']}", f"{v['opt3']}/{v['n']}", fmt(v["q3"])]
                 for m, v in rq2_rows.items()])

    print_table(f"Table 4 (RQ3): fuel-reserve requirement {safety['prop']} (optimum {safety['optimum']:.6g})"
                if safety else "Table 4 (RQ3): no finished RQ3 runs",
                ["LLM", "RQ1 networks satisfying it", "RQ3 runs satisfying it", "Mean q*", "MC rounds", "Calls"],
                [[name(m), f"{v['rq1_sat']}/{v['rq1_n']}", f"{v['opt']}/{v['n']}", fmt(v["q"]),
                  fmt(v["rounds"], 1), fmt(v["calls"], 1)] for m, v in rq3_rows.items()])

    print_table("Table 5 (RQ4): generation errors over all runs of RQ1 and RQ3",
                ["LLM", "Answers", "Invalid JSON", "Wrong shape", "Failed self-check",
                 "Accepted despite failure", "Failed requests"],
                [[name(m), v["answers"], v["invalid_json"], v["wrong_shape"], v["self_check"], v["accepted"],
                  v["failed_requests"]] for m, v in rq4_rows.items()],
                "Answers: LLM answers received; Failed self-check: answers whose network did not select the "
                "intended action\nin every check state; Accepted despite failure: such networks used after "
                "all retries.")

    total_calls = sum(r["requests"] for r in final)
    total_minutes = sum(r["seconds"] or 0 for r in final) / 60
    print(f"\nTotal: {len(final)} counted runs, {total_calls} LLM requests, {total_minutes:.0f} minutes of run time.")
    if warnings:
        print("\nCONSISTENCY WARNINGS (please inspect):")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("Consistency checks passed: re-evaluated networks match the results logged during the runs.")

    # ---- LaTeX (same captions, labels and columns as in the paper) ------------------------
    latex = latex_tables(env_rows, rq1_rows, rq1_all, rq2_rows, rq3_rows, rq4_rows, rq3, name)
    if args.latex:
        with open(args.latex, "w") as f:
            f.write(latex)
        print(f"\nLaTeX tables and the results paragraph of the introduction: {args.latex}")
    print("\n" + "=" * 78 + "\nLaTeX (copy into the paper)\n" + "=" * 78)
    print(latex)
    return 0


def latex_tables(env_rows, rq1_rows, rq1_all, rq2_rows, rq3_rows, rq4_rows, rq3, name):
    out = ["% Generated by examples/wsp_evaluate.py. Each block replaces the table with the same \\label.", ""]

    # Table 1
    rows = [f"    {row['latex_label']} & {row['d']} & {row['actions']} & {row['states']:,}".replace(",", "{,}")
            + f" & ${row['spec']['layers']} \\times {row['spec']['neurons']}$ & {tex_query(row['spec']['prop'])}"
            f" & {tex_probability(row['optimum'])} \\\\" for row in env_rows]
    out += [r"\begin{table}[t]", r"  \centering",
            r"  \caption{Environments, PCTL queries, and optimal probabilities computed by",
            r"    Storm on the MDP. $d$: number of features; $|\mathit{Act}|$: number of",
            r"    actions; $|S|$: number of MDP states; Network: hidden layers $\times$",
            r"    neurons per layer.}",
            r"  \label{tab:environments}", r"  \resizebox{\linewidth}{!}{%", r"  \begin{tabular}{lrrrcll}",
            r"    \toprule",
            r"    Environment & $d$ & $|\mathit{Act}|$ & $|S|$ & Network & PCTL query & Optimum \\",
            r"    \midrule", *rows, r"    \bottomrule", r"  \end{tabular}}", r"\end{table}", ""]

    # Table 2 (RQ1)
    def cells(row):
        c = []
        for stage in ("step1", "step2", "best"):
            st = row[stage]
            c += [tex(st["acc"], 2), tex(st["q"], 3), f"{st['opt']}/{st['n']}"]
        return " & ".join(c + [tex(row["calls"], 1)])

    rows, previous = [], None
    for row in rq1_rows:
        label = row["label"] if row["label"] != previous else ""
        previous = row["label"]
        rows.append(f"    {label} & {name(row['model'])} & {cells(row)} \\\\")
    if rq1_all:
        rows.append(r"    \midrule")
        for i, (model, row) in enumerate(rq1_all.items()):
            rows.append(f"    {'All' if i == 0 else ''} & {name(model)} & {cells(row)} \\\\")
    out += [r"\begin{table}[t]", r"  \centering",
            r"  \caption{RQ1: Accuracy and query results after each step. Acc.: mean accuracy",
            r"    over the three runs (in \%); $\bar{q}$: mean query result ($\bar{q}^*$:",
            r"    best verified result after Step~3); Opt.: runs (of 3) that reach the",
            r"    optimal probability up to $10^{-4}$; Calls: mean number of LLM calls per",
            r"    run.}",
            r"  \label{tab:rq1}", r"  \resizebox{\linewidth}{!}{%", r"  \begin{tabular}{llcccccccccc}",
            r"    \toprule",
            r"    & & \multicolumn{3}{c}{After Step 1} & \multicolumn{3}{c}{After Step 2}",
            r"      & \multicolumn{3}{c}{After Step 3} & \\",
            r"    \cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}",
            r"    Environment & LLM & Acc. & $\bar{q}$ & Opt. & Acc. & $\bar{q}$ & Opt.",
            r"      & Acc. & $\bar{q}^*$ & Opt. & Calls \\",
            r"    \midrule", *rows, r"    \bottomrule", r"  \end{tabular}}", r"\end{table}", ""]

    # Table 3 (RQ2)
    rows = [f"    {name(m)} & {v['opt2']}/{v['n']} & {v['opt3']}/{v['n']} & {tex(v['q3'], 3)} \\\\"
            for m, v in rq2_rows.items()]
    out += [r"\begin{table}[t]", r"  \centering",
            r"  \caption{RQ2: Transporter networks programmed for two jobs, model checked",
            r"    with three jobs.}",
            r"  \label{tab:rq2}", r"  \begin{tabular}{lccc}", r"    \toprule",
            r"    LLM & Optimal for 2 jobs & Optimal for 3 jobs & Mean $P(\Diamond\, \mathit{jobs\_done} = 3)$ \\",
            r"    \midrule", *rows, r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    # Table 4 (RQ3)
    rows = [f"    {name(m)} & {v['rq1_sat']}/{v['rq1_n']} & {v['opt']}/{v['n']} & {tex(v['q'], 3)}"
            f" & {tex(v['rounds'], 1)} & {tex(v['calls'], 1)} \\\\" for m, v in rq3_rows.items()]
    out += [r"\begin{table}[t]", r"  \centering",
            r"  \caption{RQ3: Fuel-reserve requirement. RQ1 networks: networks programmed only",
            r"    for $\Diamond\, \mathit{jobs\_done} = 2$ that satisfy the requirement; RQ3",
            r"    runs: runs (of 3) that satisfy it; $\bar{q}^*$ and Calls as in",
            r"    Table~\ref{tab:rq1}; Rounds: mean number of model-checking rounds.}",
            r"  \label{tab:rq3}", r"  \begin{tabular}{lccccc}", r"    \toprule",
            r"    LLM & RQ1 networks & RQ3 runs & $\bar{q}^*$ & Rounds & Calls \\",
            r"    \midrule", *rows, r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]

    # Table 5 (RQ4)
    rows = [f"    {name(m)} & {v['answers']} & {v['invalid_json']} & {v['wrong_shape']} & {v['self_check']}"
            f" & {v['accepted']} & {v['failed_requests']} \\\\" for m, v in rq4_rows.items()]
    out += [r"\begin{table}[t]", r"  \centering",
            r"  \caption{RQ4: Generation errors over all runs of RQ1 and RQ3.}",
            r"  \label{tab:rq4}", r"  \resizebox{\linewidth}{!}{%", r"  \begin{tabular}{lcccccc}",
            r"    \toprule",
            r"    LLM & Answers & Invalid JSON & Wrong shape & Failed self-check & Accepted despite failure"
            r" & Failed requests \\",
            r"    \midrule", *rows, r"    \bottomrule", r"  \end{tabular}}", r"\end{table}", ""]

    # Results paragraph of the introduction, for the LLM with the most optimal runs after Step 3.
    if rq1_all:
        best = max(rq1_all, key=lambda m: (rq1_all[m]["best"]["opt"], rq1_all[m]["best"]["acc"] or 0))
        b = rq1_all[best]
        rq2 = rq2_rows.get(best, {"opt3": "--", "n": "--"})
        rq3b = rq3_rows.get(best, {"opt": "--", "n": "--"})
        out += ["% Results paragraph of the introduction (numbers of the best LLM, i.e. the most optimal runs",
                "% after Step 3; RQ2 and RQ3 also for this LLM).",
                "Across two LLMs and three PRISM environments, the networks of the best LLM,",
                f"{name(best)}, select an optimal action in {tex(b['step1']['acc'], 1)}\\% of the states",
                f"after creation, {tex(b['step2']['acc'], 1)}\\% after dataset repair, and "
                f"{tex(b['best']['acc'], 1)}\\% after",
                f"model-checking repair, and they reach the optimal probability in {b['best']['opt']}",
                f"of {b['best']['n']} runs, compared to {b['step2']['opt']} runs without model-checking",
                "feedback. Policies programmed for two transporter jobs also complete three jobs",
                f"in {rq2['opt3']} of {rq2['n']} runs, and the pipeline programs a policy that",
                f"satisfies a fuel-reserve safety requirement in {rq3b['opt']} of {rq3b['n']}",
                "runs.", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("optimum")
    p.add_argument("--prism", required=True)
    p.add_argument("--constants", default="")
    p.add_argument("--prop", required=True)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--three-jobs-constants", required=True)
    p.add_argument("--three-jobs-prop", required=True)
    p.add_argument("--latex", default="", help="write the LaTeX tables to this file")
    args = parser.parse_args()
    if args.command == "optimum":
        optimum, states = mdp_optimum(args.prism, args.constants, args.prop)
        print(f"OPTIMUM {optimum!r} {states}")
        return 0
    return report(args)


if __name__ == "__main__":
    sys.exit(main())
