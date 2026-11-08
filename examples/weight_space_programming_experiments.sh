#!/usr/bin/env bash
# All weight-space programming (WSP) experiments of the paper.
#
#   RQ1  3 environments x 2 LLMs x 3 seeds: accuracy and query result after each pipeline step
#   RQ2  transporter networks of RQ1 (2 jobs) model checked with 3 jobs        (no LLM calls)
#   RQ3  fuel-reserve requirement P(!(passenger & fuel<4) U jobs_done=2): 2 LLMs x 3 seeds
#   RQ4  generation errors, counted from the logs of RQ1 and RQ3               (no LLM calls)
#
# At the end, examples/wsp_evaluate.py re-checks every exported network with Storm and prints
# all tables. Everything this script prints is also written to wsp_results.txt, and the
# tables are written as LaTeX (same captions and labels as in the paper) to wsp_tables.tex.
#
# Usage:
#   ./examples/weight_space_programming_experiments.sh             run all experiments, then report
#   ./examples/weight_space_programming_experiments.sh evaluate    only report the finished runs
#   DRY_RUN=1 ./examples/weight_space_programming_experiments.sh   print the planned commands only
#
# Finished runs (DONE or CREATION_FAILED in their directory) are skipped when the script is
# started again, so an interrupted batch never pays twice for the same LLM calls. Runs that
# failed for technical reasons (FAILED, e.g. Ollama unreachable) are repeated.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)

# ----------------------------------------------------------------------------------------------
# Configuration (Section "Experiments" of the paper)
# ----------------------------------------------------------------------------------------------
OLLAMA_HOST=${OLLAMA_HOST:-http://localhost:11434}   # signed-in local daemon serving *:cloud models
MODELS=(deepseek-v4-pro:cloud kimi-k3.1-pro:cloud)
SEEDS=(1 2 3)
TEMPERATURE=0
E=5             # dataset rounds (Step 2)
ALPHA=100       # target accuracy of Step 2 in percent
R=5             # model-checking rounds (Step 3): 1 created network + up to 4 repairs
RETRIES=2       # retries per LLM request (r)
TIMEOUT=1200    # seconds per LLM request
TOL=1e-4        # a run is optimal if its probability is within TOL of the optimum
MAX_STEPS=100
OUT=${OUT:-$ROOT/wsp_experiments}
RESULTS=${RESULTS:-$ROOT/wsp_results.txt}
LATEX=${LATEX:-$ROOT/wsp_tables.tex}               # LaTeX tables for the paper
PY=${PY:-python3}
EVAL=$ROOT/examples/wsp_evaluate.py

# name | label | PRISM file | constants | PCTL query | layers | neurons | reward_flag | description
RQ1_ENVS=(
  "frozen_lake|Frozen Lake|frozen_lake.prism|control=0.33,start_position=0|Pmax=? [ F pos=15 ]|1|16|1|frozen_lake_wsp.md"
  "transporter|Transporter|transporter.prism|MAX_JOBS=2,MAX_FUEL=10|Pmax=? [ F jobs_done=2 ]|2|32|0|transporter_wsp.md"
  "stock_market|Stock Market|stock_market.prism||Pmax=? [ F \"success\" ]|2|32|1|stock_market_wsp.md"
)
RQ3_ENV="transporter_safety|Transporter (safety)|transporter.prism|MAX_JOBS=2,MAX_FUEL=10|Pmax=? [ !(passenger & fuel<4) U jobs_done=2 ]|2|32|0|transporter_wsp.md"
RQ3_TOO_STRICT="Pmax=? [ !(passenger & fuel<5) U jobs_done=2 ]"   # the paper claims: cannot be met
THREE_JOBS_CONSTANTS="MAX_JOBS=3,MAX_FUEL=10"
THREE_JOBS_PROP="Pmax=? [ F jobs_done=3 ]"

declare -A OPTIMUM STATES

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

model_tag() {   # deepseek-v4-pro:cloud -> deepseek-v4-pro
  local m=${1%:cloud}
  echo "${m//[^A-Za-z0-9.-]/_}"
}

compute_optimum() {   # name prism constants prop -> fills OPTIMUM[name] and STATES[name]
  local line tag value states
  line=$("$PY" "$EVAL" optimum --prism "$ROOT/prism_files/$2" --constants "$3" --prop "$4" | grep '^OPTIMUM ' | tail -n 1) || return 1
  read -r tag value states <<< "$line"
  [[ $tag == OPTIMUM && -n $value && -n $states ]] || return 1
  OPTIMUM[$1]=$value
  STATES[$1]=$states
}

preflight() {
  local spec name label prism consts prop layers neurons rflag desc
  log "Preflight (no LLM calls)"
  for spec in "${RQ1_ENVS[@]}" "$RQ3_ENV"; do
    IFS='|' read -r name label prism consts prop layers neurons rflag desc <<< "$spec"
    [[ -f $ROOT/prism_files/$prism ]] || { log "ERROR: missing $ROOT/prism_files/$prism"; return 1; }
    [[ -f $ROOT/prism_files/$desc ]] || { log "ERROR: missing description $ROOT/prism_files/$desc"; return 1; }
    compute_optimum "$name" "$prism" "$consts" "$prop" || { log "ERROR: Storm could not check '$prop' on $prism"; return 1; }
    log "  $label: optimum of $prop = ${OPTIMUM[$name]} (MDP states: ${STATES[$name]})"
  done
  IFS='|' read -r name _ _ _ _ _ _ _ _ <<< "$RQ3_ENV"
  if ! "$PY" -c 'import sys; sys.exit(0 if float(sys.argv[1]) > 0.999 else 1)' "${OPTIMUM[$name]}"; then
    log "ERROR: the fuel-reserve requirement cannot be met (optimum ${OPTIMUM[$name]}); choose a smaller fuel threshold"
    return 1
  fi
  compute_optimum too_strict transporter.prism "MAX_JOBS=2,MAX_FUEL=10" "$RQ3_TOO_STRICT" \
    || { log "ERROR: Storm could not check '$RQ3_TOO_STRICT'"; return 1; }
  if "$PY" -c 'import sys; sys.exit(0 if float(sys.argv[1]) < 0.999 else 1)' "${OPTIMUM[too_strict]}"; then
    log "  fuel < 4 is the largest threshold that can be met (optimum with fuel < 5: ${OPTIMUM[too_strict]})"
  else
    log "  WARNING: fuel < 5 can also be met (optimum ${OPTIMUM[too_strict]}); fix the RQ3 text of the paper"
  fi
  if [[ ${DRY_RUN:-0} != 1 ]]; then
    "$PY" -c 'import sys
from ollama import Client
Client(host=sys.argv[1], timeout=30).list()' "$OLLAMA_HOST" >/dev/null 2>&1 \
      || { log "ERROR: Ollama is not reachable at $OLLAMA_HOST (sign in the daemon or set OLLAMA_HOST)"; return 1; }
    log "  Ollama reachable at $OLLAMA_HOST"
  fi
}

write_run_json() {   # dir rq order env label prism constants prop layers neurons reward_flag description model seed optimum states threshold
  "$PY" - "$@" <<'PYEOF'
import json, sys
keys = ["dir", "rq", "order", "env", "label", "prism", "constants", "prop", "layers", "neurons",
        "reward_flag", "description", "model", "seed", "optimum", "states", "threshold"]
assert len(sys.argv) - 1 == len(keys), sys.argv
run = dict(zip(keys, sys.argv[1:]))
for k in ("order", "layers", "neurons", "reward_flag", "seed", "states"):
    run[k] = int(run[k])
for k in ("optimum", "threshold"):
    run[k] = float(run[k])
with open(run.pop("dir") + "/run.json", "w") as f:
    json.dump(run, f, indent=1)
PYEOF
}

run_one() {   # rq order spec model seed
  local rq=$1 order=$2 spec=$3 model=$4 seed=$5
  local name label prism consts prop layers neurons rflag desc
  IFS='|' read -r name label prism consts prop layers neurons rflag desc <<< "$spec"
  local id="${rq}_${name}_$(model_tag "$model")_s${seed}"
  local dir="$OUT/runs/$id"
  if [[ -f $dir/DONE || -f $dir/CREATION_FAILED ]]; then
    log "skip  $id (already finished)"
    return 0
  fi

  # Stop as soon as the query is within TOL of the optimum (Pmax: optimum - TOL, Pmin: optimum + TOL).
  local kind=max threshold
  [[ $prop == Pmin* ]] && kind=min
  threshold=$("$PY" -c 'import sys
o, t = float(sys.argv[1]), float(sys.argv[2])
print(o + t if sys.argv[3] == "min" else max(0.0, o - t))' "${OPTIMUM[$name]}" "$TOL" "$kind") || return 1

  local algo="weight_space_programming_agent;model=$model;host=$OLLAMA_HOST;info_file=$ROOT/prism_files/$desc"
  algo+=";temperature=$TEMPERATURE;seed=$seed;retries=$RETRIES;timeout=$TIMEOUT;export_dir=$dir"
  local bc="all_optimal_dataset;$ROOT/prism_files/$prism;$prop;$consts"
  local cmd=("$PY" cool_mc.py
    --project_name="wsp_${rq}_${name}"
    --algorithm="$algo"
    --prism_dir="$ROOT/prism_files"
    --prism_file_path="$prism"
    --constant_definitions="$consts"
    --prop="$prop"
    --reward_flag="$rflag"
    --max_steps="$MAX_STEPS"
    --layers="$layers"
    --neurons="$neurons"
    --behavioral_cloning="$bc"
    --bc_epochs="$E"
    --accuracy_threshold="$ALPHA"
    --num_episodes="$R"
    --eval_interval=1
    --training_threshold="$threshold"
    --seed="$seed")

  if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf '%s:\n  ' "$id"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  rm -rf "$dir"
  mkdir -p "$dir"
  write_run_json "$dir" "$rq" "$order" "$name" "$label" "$ROOT/prism_files/$prism" "$consts" "$prop" \
    "$layers" "$neurons" "$rflag" "$desc" "$model" "$seed" "${OPTIMUM[$name]}" "${STATES[$name]}" "$threshold" \
    || { log "ERROR: could not write $dir/run.json"; return 1; }
  printf '%q ' "${cmd[@]}" > "$dir/command.txt"

  log "start $id"
  local start=$SECONDS
  if "${cmd[@]}" > "$dir/run.log" 2>&1; then
    echo $((SECONDS - start)) > "$dir/DONE"
    log "done  $id ($(( (SECONDS - start) / 60 )) min)"
  elif grep -q "Initial creation failed" "$dir/run.log"; then
    # The LLM wrote no valid network: a result of the experiment, not a technical failure.
    echo $((SECONDS - start)) > "$dir/CREATION_FAILED"
    log "done  $id: the LLM wrote no valid network (counted as a run without result)"
  else
    echo $((SECONDS - start)) > "$dir/FAILED"
    log "FAILED $id (see $dir/run.log); it is repeated at the next start of this script"
  fi
}

main() {
  local mode=${1:-all}
  case $mode in
    all|evaluate) ;;
    *) echo "usage: $0 [all|evaluate]"; return 2 ;;
  esac
  log "Weight-space programming experiments ($mode)"
  log "LLMs: ${MODELS[*]} | seeds: ${SEEDS[*]} | temperature $TEMPERATURE | E=$E alpha=$ALPHA% R=$R r=$RETRIES"
  log "run directories: $OUT/runs | summary: $RESULTS"

  if [[ $mode == all ]]; then
    preflight || return 1
    local order=0 spec model seed
    log "RQ1: ${#RQ1_ENVS[@]} environments x ${#MODELS[@]} LLMs x ${#SEEDS[@]} seeds"
    for spec in "${RQ1_ENVS[@]}"; do
      for model in "${MODELS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          run_one rq1 "$order" "$spec" "$model" "$seed"
        done
      done
      order=$((order + 1))
    done
    log "RQ3: fuel-reserve requirement x ${#MODELS[@]} LLMs x ${#SEEDS[@]} seeds"
    for model in "${MODELS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run_one rq3 "$order" "$RQ3_ENV" "$model" "$seed"
      done
    done
  fi

  if [[ ${DRY_RUN:-0} == 1 ]]; then
    log "DRY_RUN=1: nothing was executed"
    return 0
  fi
  log "Evaluating all finished runs with Storm (RQ1-RQ4, no LLM calls) ..."
  "$PY" "$EVAL" report --out "$OUT" --tol "$TOL" --latex "$LATEX" \
    --three-jobs-constants "$THREE_JOBS_CONSTANTS" --three-jobs-prop "$THREE_JOBS_PROP"
}

main "$@" 2>&1 | tee "$RESULTS"
exit "${PIPESTATUS[0]}"
