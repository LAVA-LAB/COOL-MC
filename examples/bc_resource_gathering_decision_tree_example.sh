#!/bin/bash
# Example: Behavioral Cloning Decision Tree for Resource Gathering with Scaling
#
# This example demonstrates how to:
# 1. Train an initial agent (PPO with raw_dataset BC) on a smaller problem (GOLD=15, GEM=15)
# 2. Use the induced_dataset to extract state-action pairs from the induced DTMC
# 3. Train a BC Decision Tree agent using those state-action pairs
# 4. Test generalization by model checking on a larger problem (GOLD=30, GEM=30)
#
# State dimensions remain the same (7 variables: x, y, gold, gem, attacked,
# required_gold, required_gem) regardless of the collection targets, making
# this suitable for transfer learning.
#
# Reference results:
#   B=200, GOLD=15, GEM=15: 24,064 states, prgoldgem=0.808
#   B=400, GOLD=30, GEM=30: 90,334 states, prgoldgem=0.865

# ============================================================================
# Pipeline: Train on GOLD=15,GEM=15 and scale to GOLD=30,GEM=30
# ============================================================================

SMALL_CONFIG="B=200,GOLD_TO_COLLECT=15,GEM_TO_COLLECT=15"
LARGE_CONFIG="B=400,GOLD_TO_COLLECT=30,GEM_TO_COLLECT=30"

# Step 1: Train initial PPO agent with raw_dataset behavioral cloning (small)
echo "=== Step 1: Training initial PPO agent with raw_dataset BC (GOLD=15, GEM=15) ==="
python cool_mc.py \
  --project_name="resource_bc_dt_small" \
  --algorithm="ppo_agent" \
  --prism_dir="../prism_files" \
  --prism_file_path="resource_gathering.prism" \
  --constant_definitions="${SMALL_CONFIG}" \
  --seed=128 \
  --layers=4 \
  --neurons=512 \
  --lr=0.0003 \
  --batch_size=64 \
  --num_episodes=1 \
  --eval_interval=1 \
  --gamma=0.99 \
  --reward_flag=1 \
  --wrong_action_penalty=0 \
  --prop="" \
  --max_steps=1000 \
  --behavioral_cloning="raw_dataset_with_anon_labels;../prism_files/resource_gathering.prism;Pmax=? [ F \"success\" ];${SMALL_CONFIG}" \
  --bc_epochs=65 \
  --deploy=1

# Step 2: Model check the trained PPO agent (small)
echo "=== Step 2: Model checking the trained PPO agent (GOLD=15, GEM=15) ==="
python cool_mc.py \
  --parent_run_id="last" \
  --project_name="resource_bc_dt_small" \
  --prism_dir="../prism_files" \
  --prism_file_path="resource_gathering.prism" \
  --constant_definitions="${SMALL_CONFIG}" \
  --task="rl_model_checking" \
  --prop="P=? [ F \"success\" ]" \
  --seed=42

# Read the project name and run_id from last_run.txt for the induced_dataset config
LAST_RUN=$(cat last_run.txt)
PROJECT_NAME=$(echo $LAST_RUN | cut -d',' -f1)
RUN_ID=$(echo $LAST_RUN | cut -d',' -f2)

echo "=== Using trained agent from: $PROJECT_NAME / $RUN_ID ==="

# Step 3: Train a BC Decision Tree agent using induced_dataset (small)
echo "=== Step 3: Training BC Decision Tree agent from induced DTMC (GOLD=15, GEM=15) ==="
python cool_mc.py \
  --project_name="resource_bc_dt_induced" \
  --algorithm=bc_decision_tree_agent \
  --prism_dir="../prism_files" \
  --prism_file_path="resource_gathering.prism" \
  --constant_definitions="${SMALL_CONFIG}" \
  --behavioral_cloning="induced_dataset;${PROJECT_NAME};${RUN_ID};P=? [ F \"success\" ];${SMALL_CONFIG}" \
  --layers=0 \
  --neurons=1 \
  --num_episodes=1 \
  --prop="Pmax=? [ F \"success\" ]" \
  --seed=42

# Step 4a: Model check the BC Decision Tree agent (small - same as training)
echo "=== Step 4a: Model checking BC Decision Tree agent (GOLD=15, GEM=15) ==="
python cool_mc.py \
  --parent_run_id="last" \
  --project_name="resource_bc_dt_induced" \
  --prism_dir="../prism_files" \
  --prism_file_path="resource_gathering.prism" \
  --constant_definitions="${SMALL_CONFIG}" \
  --task="rl_model_checking" \
  --prop="P=? [ F \"success\" ]" \
  --seed=42

# Step 4b: Model check the BC Decision Tree agent (large - scaled up!)
echo "=== Step 4b: Model checking BC Decision Tree agent (GOLD=30, GEM=30 - SCALED UP) ==="
python cool_mc.py \
  --parent_run_id="last" \
  --project_name="resource_bc_dt_induced" \
  --prism_dir="../prism_files" \
  --prism_file_path="resource_gathering.prism" \
  --constant_definitions="${LARGE_CONFIG}" \
  --task="rl_model_checking" \
  --prop="P=? [ F \"success\" ]" \
  --seed=42

echo "=== Done! ==="
echo ""
echo "Summary:"
echo "  - Trained PPO agent on B=200, GOLD=15, GEM=15 with raw_dataset BC"
echo "  - Extracted induced dataset from trained agent (~24,064 states)"
echo "  - Trained interpretable Decision Tree on induced dataset"
echo "  - Verified on GOLD=15, GEM=15 (training configuration)"
echo "  - Tested generalization on GOLD=30, GEM=30 (scaled up, ~90,334 states)"
echo ""
echo "The Decision Tree policy learned on the small problem should generalize"
echo "to the larger problem since the state dimensions are identical."
echo ""
echo "Reference optimal values:"
echo "  GOLD=15, GEM=15: prgoldgem = 0.808"
echo "  GOLD=30, GEM=30: prgoldgem = 0.865"
