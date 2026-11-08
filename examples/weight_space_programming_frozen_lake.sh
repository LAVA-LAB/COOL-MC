#!/bin/bash
# Weight-space programming (WSP) on frozen lake: an LLM writes the NN policy weights directly
# (no gradient descent), repairs them on an all-optimal-action dataset, then repairs them with
# model-checking feedback.  Property: Pmax=? [ F pos=15 ]  (best achievable: 0.82223)
#
# Usage: ./examples/weight_space_programming_frozen_lake.sh
# Overridable via environment: OLLAMA_HOST, MODEL, LAYERS, NEURONS, BC_EPOCHS, REPAIR_ROUNDS, PROJECT
set -e
cd "$(dirname "$0")/.."

# Local Ollama daemon signed in via `ollama signin` (serves *:cloud models).
# For direct cloud access use OLLAMA_HOST=https://ollama.com plus OLLAMA_API_KEY.
OLLAMA_HOST=${OLLAMA_HOST:-http://localhost:11434}
MODEL=${MODEL:-deepseek-v4-pro:cloud}
CONSTANTS="control=0.33,start_position=0"
PROP="Pmax=? [ F pos=15 ]"
LAYERS=${LAYERS:-1}                # network: 1 input (pos) -> LAYERS x NEURONS (ReLU) -> 4 actions
NEURONS=${NEURONS:-16}
BC_EPOCHS=${BC_EPOCHS:-5}          # LLM repair rounds on the dataset (0 = no dataset)
REPAIR_ROUNDS=${REPAIR_ROUNDS:-10} # model-checking repair rounds (one per episode, eval_interval=1)
PROJECT=${PROJECT:-wsp_frozen_lake}

BC=""
if [ "$BC_EPOCHS" -gt 0 ]; then
  BC="all_optimal_dataset;../prism_files/frozen_lake.prism;${PROP};${CONSTANTS}"
fi

python cool_mc.py \
  --project_name="${PROJECT}" \
  --algorithm="weight_space_programming_agent;model=${MODEL};host=${OLLAMA_HOST};info_file=../prism_files/frozen_lake_wsp.md;temperature=0.2;timeout=1200" \
  --prism_dir="../prism_files" \
  --prism_file_path="frozen_lake.prism" \
  --constant_definitions="${CONSTANTS}" \
  --reward_flag=1 \
  --max_steps=100 \
  --behavioral_cloning="${BC}" \
  --bc_epochs="${BC_EPOCHS}" \
  --layers="${LAYERS}" \
  --neurons="${NEURONS}" \
  --prop="${PROP}" \
  --training_threshold=0.822 \
  --num_episodes="${REPAIR_ROUNDS}" \
  --eval_interval=1 \
  --seed=42
