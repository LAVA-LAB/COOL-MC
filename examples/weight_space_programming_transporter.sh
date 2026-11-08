#!/bin/bash
# Weight-space programming (WSP) on the transporter (taxi) environment.
# An LLM writes the NN policy weights directly (no gradient descent), repairs them
# on an all-optimal-action dataset, and then repairs them with model-checking feedback.
#
# Usage: ./examples/weight_space_programming_transporter.sh [empty|jobs|all]   (default: all)
#   empty : Pmin=? [ F "empty" ]      never run out of fuel   (optimum 0.0)
#   jobs  : Pmax=? [ F jobs_done=2 ]  finish both jobs        (optimum 1.0)
#
# Overridable via environment: OLLAMA_HOST, MODEL, LAYERS, NEURONS, BC_EPOCHS, REPAIR_ROUNDS, PROJECT_PREFIX
set -e
cd "$(dirname "$0")/.."

# Local Ollama daemon signed in via `ollama signin` (serves *:cloud models).
# For direct cloud access use OLLAMA_HOST=https://ollama.com plus OLLAMA_API_KEY.
OLLAMA_HOST=${OLLAMA_HOST:-http://localhost:11434}
MODEL=${MODEL:-deepseek-v4-pro:cloud}
CONSTANTS="MAX_JOBS=2,MAX_FUEL=10"
LAYERS=${LAYERS:-2}                # network: 10 inputs -> LAYERS x NEURONS (ReLU) -> 6 actions
NEURONS=${NEURONS:-32}
BC_EPOCHS=${BC_EPOCHS:-5}          # LLM repair rounds on the dataset (0 = no dataset)
REPAIR_ROUNDS=${REPAIR_ROUNDS:-10} # model-checking repair rounds (one per episode, eval_interval=1)
PROJECT_PREFIX=${PROJECT_PREFIX:-wsp_transporter}

run_wsp() {
  local name="$1" prop="$2" threshold="$3"
  local bc=""
  if [ "$BC_EPOCHS" -gt 0 ]; then
    bc="all_optimal_dataset;../prism_files/transporter.prism;${prop};${CONSTANTS}"
  fi
  echo "=== WSP transporter: ${prop} (threshold ${threshold}) ==="
  python cool_mc.py \
    --project_name="${PROJECT_PREFIX}_${name}" \
    --algorithm="weight_space_programming_agent;model=${MODEL};host=${OLLAMA_HOST};info_file=../prism_files/transporter_wsp.md;temperature=0.2;timeout=1200" \
    --prism_dir="../prism_files" \
    --prism_file_path="transporter.prism" \
    --constant_definitions="${CONSTANTS}" \
    --reward_flag=0 \
    --max_steps=100 \
    --behavioral_cloning="${bc}" \
    --bc_epochs="${BC_EPOCHS}" \
    --layers="${LAYERS}" \
    --neurons="${NEURONS}" \
    --prop="${prop}" \
    --training_threshold="${threshold}" \
    --num_episodes="${REPAIR_ROUNDS}" \
    --eval_interval=1 \
    --seed=42
}

case "${1:-all}" in
  empty) run_wsp empty 'Pmin=? [ F "empty" ]' 0.01 ;;
  jobs)  run_wsp jobs  'Pmax=? [ F jobs_done=2 ]' 0.99 ;;
  all)   run_wsp empty 'Pmin=? [ F "empty" ]' 0.01
         run_wsp jobs  'Pmax=? [ F jobs_done=2 ]' 0.99 ;;
  *)     echo "usage: $0 [empty|jobs|all]"; exit 1 ;;
esac
